import os 
import torch
import pandas as pd
import torch.nn as nn 
from torch.optim import Adam
import torch.distributed as dist                                    # 用于初始化DDP训练环境
from torch.nn.parallel import DistributedDataParallel as DDP        # 导入DDP模型修饰器, 让模型能够进行DDP训练
from torch.utils.data.distributed import DistributedSampler         # 用于多进程加载数据, 作为DataLoader中sampler
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertForSequenceClassification


# 初始化进程组, linux对backend设置为 nccl
# 必须执行, 否则无法使用DDP
dist.init_process_group(backend="nccl")



# 加载数据
data_dir=""
data = pd.read_csv(data_dir)


# 创建Dataset
class TextDataset(Dataset):
    def __init__(self)->None:
        super().__init__()

        self.data = pd.read_csv(data_dir)
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]
    
    def __len__(self):
        return len(self.data)
    


# 划分数据集
# DDP是采用多进程训练数据的, 如果每个进程都进行一次划分, 那么势必会导致各训练进程间的训练数据和验证数据出现重叠的情况，会让模型过拟合
# 那么必须在多进程划分数据集中, 必须保证每一个进程间的数据划分是一致的, 而且还有保证训练和验证数据之间没有重叠的部分
# 使用generator参数来实现, 同一个随机种子来确保每次随机划分是一致的
# generator=torch.Generator().manual_seed(42)
dataset = TextDataset()
trianset, validset = random_split(dataset, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(42))


# 创建DataLoader
model_dir=""
tokenizer = BertTokenizer.from_pretrained(model_dir)


# 
def collate_func(batch):
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append[item[1]]

    inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    inputs["labels"] = torch.tensor(labels)
    return inputs



# 创建trian集和valid集的DataLoader
# 必须使用DistributedSampler()来实现让不同进程加载数据, 以实现多进程并发加载数据
train_loader = DataLoader(trianset, batch_size=10, shuffle=True, collate_fn=collate_func, sampler=DistributedSampler(trianset))
valid_loader = DataLoader(validset, batch_size=64, shuffle=False, collate_fn=collate_func, sampler=DistributedSampler(validset))


# 创建模型和优化器
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BertForSequenceClassification.from_pretrained(model_dir)


# 使用的local_rank来配置模型到底使用哪一块GPU
# 需要使用os模块获取os.environ[“LOCAL_RANK”]获取当前环境下的LOCAL_RANK值, 返回值是str, 使用时需要转为int
# 并且其他所有使用到将数据或模型转到device的代码都修改成: .to(int(os.environ["LOCAL_RANK"])
if torch.cuda.is_available():
    model = model.to(int(os.environ["LOCAL_RANK"]))

# 对模型进行DDP修饰, 使其能够进行DDP训练
model = DDP(model)   


optimer = Adam(model.parameters(), lr=2e-5)


# 多卡训练中global_rank的值是唯一的
# 即使在同一个集合中的不同容器中的global_rank也是不同的
# 所以可以通过它确定打印信息
# 注意获取global_rank的值是通过os.environ["rank"]来获取, 不是global_rank
def print_log_by_global_rank(info):
    if int(os.environ["rank"]) == 0:
        print(info)




# 训练与验证
# 注意由于不同的进程预测的正确个数不一定是相同的
# 所以需要使用all_reduce()的通信方式, 先将各个进程预测对的个数汇总起来, 然后通过汇总结果来计算累加所有进程中的预测对的总数
def evaluate():
    corr_num = 0                                # 预测正确的个数
    model.eval()                                # 开启模型测试模式
    with torch.inference_mode():                # 推理模式
        for batch in valid_loader:
            if torch.cuda.is_available():
                batch = {k:v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()}      # 将数据移动到cuda上

            output = model(**batch)             # 前向传播
            pred = torch.argmax(output.logits, dim=-1)      # 计算logits
            corr_num += (pred.long() == batch["labels"].long()).float().sum()       # 统计出预测正确的个数来计算正确率: 先将pred和batch['labels']都转成long类型, 再比较是否相等, 相等为True, 否则为False, 所以最后要将bool转成float类型(即0.0和1.0), 最后再将所有数相加就是1的个数值, 也就是预测正确的个数

    # 由于每个进程中使用部分数据进行预测的 所以才有all_reduce的通信方式来对对每个进程间的预测正确的个数的汇总
    # op是代表all_reduce内的数据进行的操作, 这里是对各进程中汇总之后在累加, 所以使用op=dist.ReduceOp.SUM
    dist.all_reduce(corr_num, op=dist.ReduceOp.SUM)
    return corr_num / len(valid_loader)                                            # 预测正确的总个数 / 测试集的总个数

                

def train(epochs=1, log_step=100):              # log_step代表每100step输出一个训练信息
    global_step = 0                             # 记录总的训练步数
    for epoch in range(epochs):
        model.train()                           # 训练模式    

        train_loader.sampler.set_epoch(epoch)   # 由于train_loader使用了sampler实现了多进程中的数据分发, 会导致每轮训练数据的顺序一致的, 所以需要使用这个按轮来打乱数据
        for batch in train_loader:              # 在训练集中训练
            if torch.cuda.is_available():   
                batch = {k:v.to(int(os.environ["LOCAL_RANK"])) for k, v in batch.items()}      # 将数据移动到cuda上
            
            optimer.zero_grad()                 # 梯度清零
            output = model(**batch)             # 将数据输入到llm中, 进行前向传播
            loss = output.loss                  # 使用loss保存output.loss
            loss.backward()                     # 反向传播
            optimer.step()                      # 反向传播后, 梯度更新

            if global_step % log_step == 0:     # 每运行100次打印一次信息
                # 由于每个进程中使用部分数据进行计算loss, 所以才有all_reduce的通信方式来对对每个进程间的loss求平均
                # op是代表all_reduce内的数据进行的操作, 这里是对各进程中的loss值来求平均, 所以使用dist.ReduceOp.AVG, 代表求平均值
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)     
                print_log_by_global_rank(f"epoch: {epoch} / {epochs + 1}, step: {global_step}, loss: {output.loss.item()}")    
            
            # 总的step数加1
            global_step += 1
            
        # 每训练完一次epoch进行验证一次
        acc = evaluate()
        print_log_by_global_rank(f"epoch: {epoch} / {epochs + 1}, acc: {acc}")    
        # 每训练完一个epcoh保存一次
        # model.save_pretrain("")



# 8. 开启训练
train()



# 注意⚠️:
"""
    
    1.运行该脚本, 必须指明一个进程数的参数 --nproc_per_node=进程数, 必须执行如下命令
      进程数根据你有多少张卡来决定, 有几张卡就设置为多少

        torchrun --nproc_per_node=进程数 当前脚本名.py

        
    2. 遗留问题
        
       多进程数据训练时, 每个进程中的batch_size时一致的, 也就是说假设训练时设定的batch_size=10, 训练进程有2个, 那么它训练的总共batch_size=20
       但是这会有一个问题: 若训练数据共有99条, 但是训练的batch_size = 50, 那么必定会有一个进程只有49条数据, 但是DDP为了保证各进程中训练数据条目数一致的
       那么会让少于50的那个进程使用padding填充为50条数据, 但这明显是不合理的, 这个问题需要等到accelerate来解决

    
    

"""
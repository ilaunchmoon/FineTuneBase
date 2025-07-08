import torch
import pandas as pd 
import torch.nn as nn 
from torch.optim import Adam
from torch.utils.data import Dataset, random_split, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, pipeline


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
dataset = TextDataset()
trianset, validset = random_split(dataset, lengths=[0.9, 0.1])



# 创建DataLoader
model_dir=""
tokenizer = BertTokenizer.from_pretrained(model_dir)


def collate_func(batch):
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append[item[1]]

    inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    inputs["labels"] = torch.tensor(labels)
    return inputs



# 创建trian集和valid集的DataLoader
train_loader = DataLoader(trianset, batch_size=10, shuffle=True, collate_fn=collate_func)
valid_loader = DataLoader(validset, batch_size=64, shuffle=False, collate_fn=collate_func)




# 创建模型和优化器
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BertForSequenceClassification.from_pretrained(model_dir, device=device)

# 启用DP多卡训练
# 注意device_ids值是list[str], 用于指定model使用那几张卡 '0', '1'...
# 当device_ids=None时, 此时默认会将当前环境下所有的卡都使用上
model = nn.DataParallel(model, device_ids=None)         

optimer = Adam(model.parameters(), lr=2e-5)




# 训练与验证
def evaluate():
    corr_num = 0                                # 预测正确的个数
    model.eval()                                # 开启模型测试模式
    with torch.inference_mode():                # 推理模式
        for batch in valid_loader:
            if torch.cuda.is_available():
                batch = {k:v.cuda() for k, v in batch.items()}      # 将数据移动到cuda上

            output = model(**batch)             # 前向传播
            pred = torch.argmax(output.logits, dim=-1)      # 计算logits
            corr_num += (pred.long() == batch["labels"].long()).float().sum()       # 统计出预测正确的个数来计算正确率: 先将pred和batch['labels']都转成long类型, 再比较是否相等, 相等为True, 否则为False, 所以最后要将bool转成float类型(即0.0和1.0), 最后再将所有数相加就是1的个数值, 也就是预测正确的个数

    return corr_num / len(valid_loader)                                            # 预测正确的总个数 / 测试集的总个数

                


def train(epochs=3, log_step=100):              # log_step代表每100step输出一个训练信息
    global_step = 0                             # 记录总的训练步数
    for epoch in range(epochs):
        model.train()                           # 训练模式    
        for batch in train_loader:              # 在训练集中训练
            if torch.cuda.is_available():   
                batch = {k:v.cuda() for k, v in batch.items()}      # 将数据移动到cuda上
            
            optimer.zero_grad()                 # 梯度清零
            output = model(**batch)             # 将数据输入到llm中, 进行前向传播
            # 模型输出中具有loss属性, 直接使用模型输出中的loss来进行反向传播, 注意由于多卡训练GPU0会将其他所有卡上计算的loss进行汇总, 此时返回给GPU0上的loss是一个向量
            # 分别代表除GPU0之外的所有卡上的loss值, 如 loss = torch.tensor([loss1, loss2, ..., lossn])
            # 但是反向传播必须要求loss值为标量, 所以对loss进行一个求均值, 从而才能保证进行反向传播
            output.loss.mean().backward()       
            optimer.step()                      # 反向传播后, 梯度更新

            if global_step % log_step == 0:     # 每运行100次打印一次信息
                # 使用到loss.mean()来替换loss
                print(f"epoch: {epoch} / {epochs + 1}, step: {global_step}, loss: {output.loss.mean().item()}")    
            
            # 总的step数加1
            global_step += 1
            
        # 每训练完一次epoch进行验证一次
        acc = evaluate()
        print(f"epoch: {epoch} / {epochs + 1}, acc: {acc}")    
        # 每训练完一个epcoh保存一次
        # model.save_pretrain("")



# 8. 开启训练
train()


# 9. 预测模型
sen = "好吃"
id2_label = {0: "negtive", 1: "postive"}
model.eval()
with torch.inference_mode():
    inputs = tokenizer(sen, return_tensors="pt")
    inputs = {k:v.cuda() for k, v in inputs.items()}
    logits = model(**input).logits
    pred = torch.argmax(logits, dim=-1)
    print(f"Input: {sen} \n, 预测结果: {id2_label.get(pred.item())}")


# 10. 使用pipeline来预测
model.config.id2label = id2_label           # 设置标签映射
pipe = pipeline("text_classification", model=model, tokenizer=tokenizer, device=0)  # 注意有cuda才设置为0









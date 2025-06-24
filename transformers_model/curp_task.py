"""
    使用一个酒店评论来进行情感二分类任务的训练
"""
import torch
import pandas as pd
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


device = "cuda" if torch.cuda.is_available() else "cpu"

# 0. 加载tokenizer和model
model = AutoModelForSequenceClassification.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model/rbt3").to(device)
tokenizer = AutoTokenizer.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model/rbt3")


# 1. 加载数据集
data = pd.read_csv("/Users/icur/CursorProjects/FineTuneBase/data/Corpdb/ChnSentiCorp_htl_all.csv")
# print(data[:2])
# print(len(data))


# 1.1 数据预处理: 去空值的数据
process_data = data.dropna()
# print(len(process_data))


# 2. 将原始数据转换为Dataset下的数据, 便于输入到模型
class ChnDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv("/Users/icur/CursorProjects/FineTuneBase/data/Corpdb/ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]

    def __len__(self):
        return len(self.data)

# 2.1 测试
dataset = ChnDataset()
# for i in range(5):
#     print(dataset[i])


# 3. 划分测试集、训练集、验证集
train_dataset, valid_dataset = random_split(dataset=dataset, lengths=[0.9, 0.1])
# print(len(train_dataset))
# print(len(valid_dataset))



# 4. 将数据集处理成能够按照批次输入
train_dataset_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_dataset_loader = DataLoader(valid_dataset, batch_size=4, shuffle=True)
# 打印出来发现: label转为了tensor形式(数字形式会直接转为tensor形式), 但是review还是文本形式, 那是因为还没有使用tokenizer进行转换
# print(next(enumerate(train_dataset_loader)))    


def collate_fn(batch):
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])           # 把review对应的添加到texts中
        labels.append(item[1])          # 把labels对应的添加到labels中
    
    # 进行的token化
    inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")        # 注意这里是一个DatasetDict包含: input_ids, attention_mask, token_type_ids 三个键值对
    inputs["labels"] = torch.tensor(labels)                                                 # 所以可以使用inputs["labels"]给inputs添加一个键值对, 使用上面labels直接使用torch转换赋值即可
    return inputs

# 4.1 将review对应的文本进行tokenizer处理, 使用collate_fn来按批次处理
train_dataset_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
valid_dataset_loader = DataLoader(valid_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
print(next(enumerate(train_dataset_loader))) 



# 5. 创建优化器
optimer = Adam(model.parameters(), lr=2e-5)     # 将模型参数和lr设置好


# 6. 训练和验证
def evaluate():
    corr_num = 0                                # 预测正确的个数
    model.eval()                                # 开启模型测试模式
    with torch.inference_mode():                # 推理模式
        for batch in valid_dataset:
            if torch.cuda.is_available():
                batch = {k:v.cuda() for k, v in batch.items()}      # 将数据移动到cuda上

            output = model(**batch)             # 前向传播
            pred = torch.argmax(output.logits, dim=-1)      # 计算logits
            corr_num += (pred.long() == batch["labels"].long()).float().sum()       # 统计出预测正确的个数来计算正确率: 先将pred和batch['labels']都转成long类型, 再比较是否相等, 相等为True, 否则为False, 所以最后要将bool转成float类型(即0.0和1.0), 最后再将所有数相加就是1的个数值, 也就是预测正确的个数

    return corr_num / len(valid_dataset)                                            # 预测正确的总个数 / 测试集的总个数

                


def train(epochs=3, log_step=100):              # log_step代表每100step输出一个训练信息
    global_step = 0                             # 记录总的训练步数
    for epoch in range(epochs):
        for batch in train_dataset:             # 在训练集中训练
            if torch.cuda.is_available():
                batch = {k:v.cuda() for k, v in batch.items()}      # 将数据移动到cuda上
            
            optimer.zero_grad()                 # 梯度清零
            output = model(**batch)             # 将数据输入到llm中, 进行前向传播
            output.loss.backward()              # 模型输出中具有loss属性, 直接使用模型输出中的loss来进行反向传播
            optimer.step()                      # 反向传播后, 梯度更新

            if global_step % log_step == 0:     # 每运行100次打印一次信息
                print(f"epoch: {epoch} / {epochs + 1}, step: {global_step}, loss: {output.loss.item()}")    
            
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







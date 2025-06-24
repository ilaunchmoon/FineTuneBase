"""
    使用datasets来进行数据加载、预处理、批加载、最后结合torch.util.data DataLoader转为可以输入模型中的数据
"""
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 加载模型和tokenizer
model = AutoModelForSequenceClassification.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model/bert-base-chinese")
tokenizer = AutoTokenizer.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model/bert-base-chinese")


# 2. 加载数据
train_dataset = load_dataset("/Users/icur/CursorProjects/FineTuneBase/data/Corpdb", split="train")
train_dataset = train_dataset.filter(lambda x: x["review"] is not None)


# 3. 划分数据
datasets = train_dataset.train_test_split(test_size=0.1)        # 分割出了训练和验证数据

# 4. 创建能够输入到LLM的DataLoader
def process_data(examples):
    tokenized_examples = tokenizer(examples["review"], max_length=128, truncation=True)
    tokenized_examples["labels"] = tokenized_examples["label"]
    return tokenized_examples

# mapping并进行删除原始字段: 这里还是list类型, 不是tensor类, 则需要结合DataLoader转为tensor类型
tokenized_datasets = datasets.map(process_data, batched=True, remove_columns=datasets.column_names)   # tokenized_datasets 具有两部分: train和valid, 都是DatasetDict
 

# 5. 创建能够输入到LLM中的DataLoader
train_set, valid_set = tokenized_datasets["train"], tokenized_datasets["test"]
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=DataCollatorWithPadding(tokenizer))
valid_loader = DataLoader(valid_set, batch_size=4, shuffle=True, collate_fn=DataCollatorWithPadding(tokenizer))

# next(enumerate(train_loader))[1] # 输出一条看看是否为正确能够输入到LLM中的格式


# 5. 创建优化器
optimer = Adam(model.parameters(), lr=2e-5)     # 将模型参数和lr设置好


# 6. 训练和验证
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

    return corr_num / len(valid_set)                                            # 预测正确的总个数 / 验证集的总个数

                


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



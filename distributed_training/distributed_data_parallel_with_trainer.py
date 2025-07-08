"""
    全部使用transformers中trainer, 不用额外配置任何参数, 会自动适配DDP来训练

"""



import torch
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments



# 1. 加载模型和tokenizer
model = AutoModelForSequenceClassification.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model/bert-base-chinese")
tokenizer = AutoTokenizer.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model/bert-base-chinese")


# 2. 加载数据
train_dataset = load_dataset("/Users/icur/CursorProjects/FineTuneBase/data/Corpdb", split="train")
train_dataset = train_dataset.filter(lambda x: x["review"] is not None)


# 3. 划分数据
datasets = train_dataset.train_test_split(test_size=0.1, seed=42)        # 分割出了训练和验证数据, 必须指定随机种子, 防止每个进程间因为打乱不一致, 而导致不同进程间存在训练和验证集重复现象

# 4. 创建能够输入到LLM的DataLoader
def process_data(examples):
    tokenized_examples = tokenizer(examples["review"], max_length=128, truncation=True)
    tokenized_examples["labels"] = examples["label"]
    return tokenized_examples

# mapping并进行删除原始字段: 这里还是list类型, 不是tensor类, 则需要结合DataLoader转为tensor类型
tokenized_datasets = datasets.map(process_data, batched=True, remove_columns=datasets["train"].column_names)   # tokenized_datasets 具有两部分: train和valid, 都是DatasetDict
 




# 5 创建评估函数
acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = predictions.argmax(axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    acc.update(f1)
    return acc


# 6. 创建TrainingArguments 和 Trainer
# 自行配置其他必要参数
training_args = TrainingArguments(
                                  output_dir="/Users/icur/CursorProjects/FineTuneBase/outputs/checkponits",
                                  num_train_epochs=1,
                                  per_device_eval_batch_size=10,
                                  per_device_train_batch_size=5,
                                  save_safetensors=True,
                                  save_strategy="epoch")


trainer = Trainer(
                  model=model, 
                  args=training_args, 
                  train_dataset=tokenized_datasets["train"], 
                  eval_dataset=tokenized_datasets["test"],
                  data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                  compute_metrics=eval_metric
                  )

# 7. 启动训练
trainer.train()

# 8. 启动模型评估
trainer.evaluate()      # tokenized_datasets["train"] 可以知道评估数据集为训练集等任何数据集


# 9. 模型预测
trainer.predict(tokenized_datasets["test"])


"""
    
    1.运行该脚本, 必须指明一个进程数的参数 --nproc_per_node=进程数, 必须执行如下命令
      进程数根据你有多少张卡来决定, 有几张卡就设置为多少

        torchrun --nproc_per_node=进程数 当前脚本名.py

"""
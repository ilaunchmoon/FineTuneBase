"""
    优化器显存的优化

        使用显存使用更小的优化器来进行训练, 如adafactor优化器

        实现方法:  optim=adafactor

        注意: 不会影响llm训练效果, 只会增加更多的训练时间

"""

import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding


# 加载数据集, 当前只加载该数据集下的训练集: 仅仅设置为数据所在的根目录
dataset = load_dataset("/Users/icur/CursorProjects/FineTuneBase/data/Corpdb/", split="train")
dataset = dataset.filter(lambda x : x["review"] is not None)

# 划分数据集
datasets = dataset.train_test_split(test_size=0.1)      # 此时从上面dataset划分出训练和测试数据集


# 数据预处理: 主要是结合tokenizer转变为可以输入到LLM的数据格式
tokenizer = AutoTokenizer.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model/bert-base-chinese")
def process_data(example):
    tokenized_data = tokenizer(example["review"], max_length=128, padding="max_length", truncation=True)
    tokenized_data["labels"] = example["label"]
    return tokenized_data

tokenized_data = datasets.map(process_data, batched=True, remove_columns=datasets["train"].column_names)



# 创建模型
model = AutoModelForSequenceClassification.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model/bert-base-chinese")


# 创建评估函数
acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = predictions.argmax(axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    acc.update(f1)
    return acc 


# 配置训练参数
train_args = TrainingArguments(
    output_dir="/Users/icur/CursorProjects/FineTuneBase/outputs/bert_base_chinese_down_tasks",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,         # 每32个批次进行一次梯度更新
    gradient_checkpointing=True,            # 激活检查点
    optim="adafactor",                      # 使用显存更少的优化器
    logging_steps=500,
    num_train_epochs=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    metric_for_best_model="f1",
    load_best_model_at_end=True 
)


# 定义训练器
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=eval_metric
)


# 模型开启训练
trainer.train()


# 开启验证
trainer.eval()


# 开启测试
trainer.predict(tokenized_data["test"])

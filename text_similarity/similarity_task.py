"""
    针对输入： 文本对和相似度标签的文本相似度任务, 交互式方法, 使用单个模型

"""

import torch
import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding, pipeline


# 加载数据集
# 注意由于这里为json文件，不是hf平台在线下载的数据集文件, 要指明.json和本地加载的路径
dataset_files = "/Users/icur/CursorProjects/FineTuneBase/data/train_pair_1w.json"
dataset = load_dataset("json", data_files=dataset_files, split="train")

# 划分验证和训练集(占0.2)
datasets = dataset.train_test_split(test_size=0.2)


# 数据预处理
model_dir = "/Users/icur/CursorProjects/FineTuneBase/model_cache/chinese-macbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_dir)

def process_func(examples):
    tokenized_examples = tokenizer(examples["sentence1"], examples["sentence2"], max_length=128, truncation=True)
    tokenized_examples["labels"] = [int(label) for label in examples["label"]]      # 将examples中label字段转成int类型, 然后添加给tokenized_examples的字段labels
    return tokenized_examples


# 对数据进行batch处理并删除数据集中原始的字段: Sentence1 Sentence1 label
tokenized_datasets = datasets.map(process_func, batched=True, remove_columns=datasets["train"].column_names)
# print(tokenized_datasets["train"][0])



# 创建模型
model = AutoModelForSequenceClassification.from_pretrained(model_dir)


# 评估函数
acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def eval_metric(eval_predict):
    predictions, labels = eval_predict
    # 确保predictions和labels不是None
    if predictions is None or labels is None:
        return {"accuracy": 0.0, "f1": 0.0}
    predictions = predictions.argmax(axis=-1)
    
    # 计算accuracy - 注意参数名是references而不是reference
    acc = acc_metric.compute(predictions=predictions, references=labels)
    
    # 计算f1 - 确保参数名一致
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    
    # 合并结果
    result = {}
    result.update(acc)
    result.update(f1)
    
    return result



# 配置训练参数
training_args = TrainingArguments(
    output_dir="/Users/icur/CursorProjects/FineTuneBase/outputs/text_similarity",
    per_device_train_batch_size=10,
    per_device_eval_batch_size=20,
    num_train_epochs=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=500,
    load_best_model_at_end=True 
)


# 创建训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=eval_metric
)


# 开启训练
trainer.train()


# 开启评估
trainer.evaluate(tokenized_datasets["test"])


# 开启测试
model.config.id2label = {0: "不相似", 1: "相似"}
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

# 需指定句子对的输入, 需要将它们使用一个字典的方法构成一个pair作为输入
# 字典中的键必须为: text和text_pair
pipe({"text": "喜欢你", "text_pair": "好喜欢你"})

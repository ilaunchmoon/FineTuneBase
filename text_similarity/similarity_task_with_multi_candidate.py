"""
    针对输入： 输入文本和多个候选文本, 找出候选文本中与输出文本最为相似的候选文本, 有点类似多项选择任务, 交互式方法, 使用单个模型

    解决方案

        假设每个输入文本需要与2个候选文本进行判断, 判断出两个候选文本中与输入文本最为相似的文本, 可按如下

            训练数据格式如下:

                Sentence     candidation1,candidation2      labels

            数据预处理：

                CLS1     Sentence     SEP     candidation1    SEP
                CLS1     Sentence     SEP     candidation2    SEP

            输出：

                CLS1输出一个相似度分数, 即计算了 Sentence 与 candidation1的相似度分数score1
                CLS2输出一个相似度分数, 即计算了 Sentence 与 candidation2的相似度分数score2

                最终, 针对这两个相似度分数进行argmax()输出最早两个获选文本和输入文本最为相似的文本
                其实这里的argmax操作就是对两个相似度得分找出最大的相似度得分的索引, 通过这个索引到获选文本中对应该索引位置的候选文本, 这个获选文本就是与输入文本最为相似的文本


    评估函数

        由于最终输出是相似度得分, 那么损失函数可以使用均方差不能使用交叉熵, 所以labels必须为float类型

"""


import torch
import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding, pipeline


# 加载数据集
# 注意由于这里为json文件, 不是hf平台在线下载的数据集文件, 要指明.json和本地加载的路径
dataset_files = "/Users/icur/CursorProjects/FineTuneBase/data/train_pair_1w.json"
dataset = load_dataset("json", data_files=dataset_files, split="train")

# 划分验证和训练集(占0.2)
datasets = dataset.train_test_split(test_size=0.2)


# 数据预处理
model_dir = "/Users/icur/CursorProjects/FineTuneBase/model_cache/bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_dir)

def process_func(examples):
    tokenized_examples = tokenizer(examples["sentence1"], examples["sentence2"], max_length=128, truncation=True)
    tokenized_examples["labels"] = [float(label) for label in examples["label"]]      # 将examples中label字段转成float类型, 然后添加给tokenized_examples的字段labels
    return tokenized_examples


# 对数据进行batch处理并删除数据集中原始的字段: Sentence1 Sentence1 label
tokenized_datasets = datasets.map(process_func, batched=True, remove_columns=datasets["train"].column_names)
print(tokenized_datasets["train"][0])



# 创建模型
model = AutoModelForSequenceClassification.from_pretrained(model_dir, numbel_label=1)       # 预测的相似度得分, 所以输出维度为1, 此时该模型任务头自动会任务回归任务


# 评估函数
acc_metirc = evaluate.load("accuracy")
f1_metirc = evaluate.load("f1")

def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = [int(p > 0.5) for p in predictions]
    labels = [int(label) for label in labels]               # 训练的时候由于使用均方差损失, label是float类型, 现在需要改成int类型
    acc = acc_metirc.compute(predictions=predictions, reference=labels)
    f1 = f1_metirc.compute(predictions=predictions, references=labels)
    acc.update(f1)
    return acc



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
response = pipe({"text": "喜欢你", "text_pair": "好喜欢你"}, funcation_to_appley=None)
response["label"] = "相似" if response["score"] > 0.5 else "不相似"

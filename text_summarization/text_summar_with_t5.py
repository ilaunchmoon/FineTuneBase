"""
    文本摘要任务

    模型: 使用编码器-解码器架构的T5模型

    数据预处理:

        input和labels分开处理, labels的最后一个一定是结束特殊标记EOS, labels还需作为解码器的输入

            编码器输入      解码器输入 token by token
        1 2 3 4 5 eos           bos 6 7 8 9 eos

                                    bos 7 8 9 eos

                                    解码器自回归输入和输出


    模型:  AutoModelForSeq2SeqLM, 最典型的为T5模型



"""     
import torch
import numpy as np 
from rouge_chinese import Rouge
from datasets import Dataset
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, pipeline


# 下载模型
snapshot_download(
    repo_id="Langboat/bloom-389m-zh",  # 模型ID
    local_dir="/Users/icur/CursorProjects/FineTuneBase/model_cache/mengzi-t5"
)


# 加载数据集
data_dir = "/Users/icur/CursorProjects/FineTuneBase/data/nlpcc_2017"
ds = Dataset.load_from_disk(data_dir)

# 数据分割
ds = ds.train_test_split(0.1, seed=42)



# 数据预处理
model_dir = "/Users/icur/CursorProjects/FineTuneBase/model_cache/mengzi-t5"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
def process_func(examples):
    contents = ["摘要生成: \n" + e for e in examples["contents"]]          # 给原始contents字段前面添加一个 "摘要生成: \n", 再进行嵌入, 作为输入
    inputs = tokenizer(contents, max_length=384, truncation=True)         # 将contents字段嵌入
    labels = tokenizer(text_target=examples["title"], max_length=64, truncation=True)    # 将原始title字段对应的值嵌入, 作为labels
    inputs["labels"] = labels["input_ids"]                                               # 将labels的内容添加给inputs, 
    return inputs

tokenized_ds = ds.map(process_func, batched=True)


# 创建模型
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)


# 创建评估函数
rouge = Rouge()

def compute_metric(pred):
    prediction, labels = pred
    # 由于上面输出的是logits, 而使用Rouge-n进行评估是需要使用文本token, 所以需要解码
    decode_preds = tokenizer.batch_decode(prediction, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)               # 将labels中pad去除
    decode_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)        # 再将labels进行解码
    decode_preds = [" ".join(p) for p in decode_preds]                              # 由于训练数据是中文, 所以需要使用空格来做分词
    decode_labels = [" ".join(l) for l in decode_labels]                            # 由于训练数据是中文, 所以需要使用空格来做分词

    # 使用rouge计算评估score
    scores = rouge.get_scores(decode_preds, decode_labels, avg=True)                # avg代表取平均值

    # 分别获取rouge-1、rouge-2、rouge-l分数返回
    return {
        "rouge-1": scores["rouge-1"]["f"],
        "rouge-2": scores["rouge-1"]["f"],
        "rouge-l": scores["rouge-l"]["f"]
    }



# 配置训练参数: Seq2SeqTrainingArguments 和 TrainingArguments本质是一样的
args = Seq2SeqTrainingArguments(
    output_dir="/Users/icur/CursorProjects/FineTuneBase/outputs/t5-summary",
    per_device_train_batch_size=10,
    per_device_eval_batch_size=20,
    gradient_accumulation_steps=2,
    logging_steps=8,
    save_strategy="epoch",
    eval_strategy="epoch",
    metric_for_best_model="rouge-l",
    predict_with_generate=True                  # 必须将该参数设置为True, 否则无法做评估
)


# 创建训练器
# Seq2SeqTrainer 本质和Trainer没有区别
trainer = Seq2SeqTrainer(
    args=args,
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    compute_metrics=compute_metric,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer)
)



# 开启训练
trainer.train()



# 模型推理
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
response = pipe("摘要生成: \n" + ds["test"][-1]["content"], max_length=100, do_example=True)         # 使用测试集数据中的content直接来测试模型训练效果
print(response)

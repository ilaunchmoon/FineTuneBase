"""
    NER任务
    
    数据集: peoples_daily_ner

    模型: hfl/chinese-macbert-base
"""
import numpy as np
import evaluate
from huggingface_hub import snapshot_download
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification, pipeline


# 加载数据集
peoples_daily_ner_load = load_dataset("peoples_daily_ner", cache_dir="/Users/icur/CursorProjects/FineTuneBase/data_cache/ner")

# 加载模型
# snapshot_download(
#     repo_id="hfl/chinese-macbert-base",  # 模型ID
#     local_dir="/Users/icur/CursorProjects/FineTuneBase/model_cache/chinese-macbert-base"
# )

# 查看数据信息和结构等
# print(peoples_daily_ner_load)   # 输出DatasetDict, 包含3个数据集: train、validation、test
# print(peoples_daily_ner_load["train"].features)     # 查看train训练数据集的features:  {'id': Value(dtype='string', id=None), 'tokens': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None), 'ner_tags': Sequence(feature=ClassLabel(names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC'], id=None), length=-1, id=None)}
# print(peoples_daily_ner_load["train"].features["ner_tags"].feature.names)      # 查看ner的标签: ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']


# 获取标签
labels_list = peoples_daily_ner_load["train"].features["ner_tags"].feature.names


# token化
tokenizer = AutoTokenizer.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model_cache/chinese-macbert-base")
tokenized_tokens_data = tokenizer(peoples_daily_ner_load["train"][0]["tokens"])        # 注意: ner的数据都是token级的输入, 所以它的词嵌入原料就是token级构成的序列
# print(tokenized_tokens_data)   # 由于NER的数据都是以每个token作为一个单独的序列, 导致tokenizer也是以单个token作为一个序列作为输入, 需要将输入序列修改成单个句子作为输入
tokenized_tokens_data = tokenizer(peoples_daily_ner_load["train"][0]["tokens"], is_split_into_words=True) # 使用 is_split_into_words=True 设置将一个句子作为整体进行嵌入，而不是使用每一个token进行嵌入
# print(tokenized_tokens_data)   # 此时就是将一个个句子作为tokenizer词嵌入的原料

# 针对英文的情况: 分词的过程可能会将单词分成多个子词, 那么此时标记NER任务的标签需要表明哪些子分词是来自同一个词
# 会被分成5个分词, 第一个单词分成4个分词, 第二个单词分词1个分词, 如果直接将这样的分词结果进行NER标注, 那么势必无法标记出哪些分词是属于同一个单词的标签, 所以需要结合 word_ids()来标记是否来自分词是否来自同一个分词
# word_ids()会将每一个分词的来源位置表明出来
print(tokenizer("interesting word"))       

# word_ids()会将每一个分词的来源位置表明出来
print(tokenizer("interesting word").word_ids())  # [None, 0, 0, 0, 0, 1, None] None代表CLS和SEP特殊token,  0, 0, 0, 0代表这些token来自输入文本的第0个单词, 1代表该token来自输入文本的第1个token


def process_data(example):
    # example是完整的数据集, 包含train、test、validation, 注意这里的不能使用return_tensors="pt"来返回tensor形式
    tokenized_data = tokenizer(example["tokens"], max_length=128, truncation=True, is_split_into_words=True, )        # 仅针对输入数据的tokens字段进行tokenize化
    labels = []
    for index, label in enumerate(example["ner_tags"]):             # 针对完整数据中的ner_tags字段进行处理
        word_ids = tokenized_data.word_ids(batch_index=index)       # 获取当前输入token嵌入之后的数据中的word_idx位置信息
        label_ids = []
        for word_id in word_ids:
            if word_id is None:                         # 如果word_id为None, 说明当前token是特殊标签CLS或SEP等, 所以它的标签值设置为-100, 这样在计算交叉熵时就会被自动忽略
                label_ids.append(-100)                  #
            else:
              label_ids.append(label[word_id])        
        labels.append(label_ids)                        # 将当前example["tokens"]对应的所有ner标签都填加到labels中
    return tokenized_data   



# 数据预处理
tokenized_datasets = peoples_daily_ner_load.map(process_data, batched=True, remove_columns=peoples_daily_ner_load["train"].column_names)
# print(tokenized_datasets["train"][0])



# 创建评估器
seqeval = evaluate.load("seqeval")

def eval_metric(pred):
    predictions, labels = pred
    predictions = np.argmax(axis=-1)            # 由于predictions不是torch类型而是list类型, 则使用numpy进行求解
    
    # 将NER的数字标签转为对应的字符标签
    # 预测标签
    true_predictions = [
        [labels_list[p] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # 数字标签转为对应的字符标签
    # 真实标签
    true_labels = [
        [labels_list[l] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    metric_rest = seqeval.compute(predictions=true_predictions, references=true_labels, mode="strict", scheme="IOB2")   # scheme代表NER的标注方式

    return {
        "f1": metric_rest["overall_f1"]
    }
    


# 创建模型
num_labels = len(peoples_daily_ner_load["train"].features["ner_tags"].feature.names) 
model = AutoModelForTokenClassification.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model_cache/chinese-macbert-base", num_labels = num_labels)
# print(model)



# 创建训练参数
training_args = TrainingArguments(
    output_dir="/Users/icur/CursorProjects/FineTuneBase/outputs/ner",
    per_device_train_batch_size=10,
    per_device_eval_batch_size=20,
    num_train_epochs=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="f1",
    logging_steps=500,
    load_best_model_at_end=True 
)


# 创建训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=eval_metric,
    data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer)
)



# 开启训练
trainer.train()


# 开启验证
trainer.evaluate(eval_datasets=tokenized_datasets["test"])



# 开始预测
# 设置id2lables
# 通过 num_labels 设置的类别, 它只会是LABEL0 到 LABEL6, 所以需要建立该任务到模型类别的映射
model.config.id2label = {idx: label for idx, label in enumerate(labels_list)}

# aggregation_strategy="simple"代表最后对输入序列中每个实体进行输出, 而不是针对每一个token进行实体输出
ner_pipe = pipeline("token_classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
response = ner_pipe("特朗普在北京访问")

ner_rest = {}
x = "特朗普在北京访问"
for r in response:
    if r["entity_group"] not in ner_rest:
        ner_rest[r["entity_group"]] = []
    ner_rest[r["entity_group"]].append(x[r["start"]: r["end"]])


print(ner_rest)


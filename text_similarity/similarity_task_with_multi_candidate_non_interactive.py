"""
    针对输入： 输入文本和多个候选文本, 找出候选文本中与输出文本最为相似的候选文本, 有点类似多项选择任务, 非交互式方法, 使用两个模型
    
    交互式方法缺点: 输入一个文本从获选文本中选出一个最为相似的文本输出, 假设获选文本中具有100w条数据, 每进行一次模型推理需要100w次相似度比较, 效率极低

    改进方法：本次介绍的双模型方法(向量匹配方法), 有点类似RAG的向量索引阶段操作

    解决方法:

        将所有获选文本使用Bert模型(LLM)进行编码, 编码具有语义信息的稠密向量, 并存放到向量数据库

        将用户的提问使用同一个模型(也可以不是同一个模型)进行编码, 得到编码稠密向量, 然后使用户的提问向量使用相似度度量方法与向量数据库进行匹配, 得到最为相似获选文本向量, 最后将最为相似的获选文本向量解码为文本输出即可


    具体做法：

        用户文本输入Bert模型得出文本向量A, 候选文本输入Bert模型得到文本向量B
        训练目标尽可能让A和B的相似度分数接近1

    
    数据预处理：


        [CLS] SentenceA [SEP] 
        [CLS] SentenceB [SEP]           即 [batch_size, 2, seq_dim]

        都是处理成一对对的文本输入


    模型损失函数(CosinEmbeddingLoss)

        loss(x, y) = 1 - cos(x1, x2) if y = 1 即当文本相似时
        loss(x, y) = max(0, cos(x1, x2)- margin) if y=-1 即文本不相似时,  其中margin为边界值, 即小于某个值为不相似, 否则相似


"""


import torch
import evaluate
from typing import Optional
from datasets import load_dataset
from torch.nn import CosineEmbeddingLoss, CosineSimilarity
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification, BertModel, PretrainedConfig, BertPreTrainedModel, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding, pipeline


# 加载数据集
# 注意由于这里为json文件, 不是hf平台在线下载的数据集文件, 要指明.json和本地加载的路径
dataset_files = "/Users/icur/CursorProjects/FineTuneBase/data/train_pair_1w.json"
dataset = load_dataset("json", data_files=dataset_files, split="train")

# 划分验证和训练集(占0.2)
datasets = dataset.train_test_split(test_size=0.2)


# 数据预处理
model_dir = "/Users/icur/CursorProjects/FineTuneBase/model_cache/chinese-macbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_dir)

def process_func(examples):
    sentences = []  
    labels = []

    # 遍历每一条数据
    for sen1, sen2, label in zip(examples["sentence1"], examples["sentence2"],  examples["label"]):
        sentences.append(sen1)
        sentences.append(sen2)
        labels.append(1 if int(label)== 1 else -1)          # 修改一下标签, 即如果两个数据相似则为1, 否则为-1

    # tokenized_examples: input_ids, token_type_ids, attention_mask
    tokenized_examples = tokenizer(sentences, max_length=128, truncation=True, padding="max_length")

    # 现在需要将tokenized_examples将两个两个组合成一对作为输入
    """
        原本为:
            {
                input_ids: [1, 2, 3, 4],
                token_type_ids: [3, 2, 1, 6],
                attention_mask: [5, 4, 5, 3]
            }

        现在为:

            {
                input_ids: [[1, 2], [3, 4]],
                token_type_ids: [[3, 2], [1, 6]],
                attention_mask: [[5, 4], [5, 3]]
            }
    
    """
    tokenized_examples = {k:[v[i:i+2] for i in range(0, len(v), 2)] for k, v in tokenized_examples.items()}
    tokenized_examples["labels"] =  examples["label"]      # 给添加tokenized_examples的字段labels
    return tokenized_examples


# 对数据进行batch处理并删除数据集中原始的字段: Sentence1 Sentence1 label
tokenized_datasets = datasets.map(process_func, batched=True, remove_columns=datasets["train"].column_names)
print(tokenized_datasets["train"][0])



# 创建模型
# 由于双模型组合没有现成得模型可以用, 需要自定义
class DualModels(BertPreTrainedModel):  # 继承BertPreTrainedModel主要时为了使用.from_pretrained()方法
    def __init__(self, config:PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = BertModel(config)
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1. 获取sentenceA和sentenceB的输入
        senA_input_ids, senB_input_ids = input_ids[:0], input_ids[:1]
        senA_attention_mask, senB_attention_mask = attention_mask[:0], attention_mask[:1]
        senA_token_type_ids, senB_token_type_ids = token_type_ids[:0], token_type_ids[:1]
        
        # 2.获取sentenceA和sentenceB的embedding表示向量, 向量表示就是bert模型的输出
        senA_outputs = self.bert(
            senA_input_ids,
            attention_mask=senA_attention_mask,
            token_type_ids=senA_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        senA_pooled_output = senA_outputs[1]               # 即为A的最终Bert编码表示[batch, dim]

        senB_outputs = self.bert(
            senB_input_ids,
            attention_mask=senB_attention_mask,
            token_type_ids=senB_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        senB_pooled_output = senB_outputs[1]               # 即为B的最终Bert编码表示[batch, dim]


        # 3. 计算相似度作为评估方法
        # 原本的分类模型会在bert输出后计算logits
        cos = CosineSimilarity(senA_outputs, senB_outputs)  # [batch]   

        # 计算相似度
        loss = None
        if labels is not None:
            loss 





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

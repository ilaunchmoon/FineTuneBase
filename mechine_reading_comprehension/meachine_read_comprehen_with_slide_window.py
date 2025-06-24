import collections
import numpy as np
from datasets import load_from_disk
from cmrc_eval import evaluate_cmrc
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments, DefaultDataCollator, pipeline


# 加载数据集
datasets = load_from_disk("/Users/icur/CursorProjects/FineTuneBase/data/cmrc2018")
# print(datasets)


# 数据预处理
tokenizer = AutoTokenizer.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model_cache/chinese-macbert-base")
sample_dataset = datasets["train"].select(range(10))
# 使用 return_overflowing_tokens=True设置为滑动窗口, 并使用stride来控制重叠部分
tokenized_examples = tokenizer(text=sample_dataset["question"],
                               text_pair=sample_dataset["context"],
                               return_offsets_mapping=True,
                               return_overflowing_tokens=True,
                               stride=128,
                               max_length=384, 
                               truncation="only_second",
                               padding="max_length")

# print(tokenized_examples.keys())     # 输出 dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'overflow_to_sample_mapping'])
# print(tokenized_examples["overflow_to_sample_mapping"])     # [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9] 可以看到将10条数据切成29条具有重叠部分的数据
print(datasets["train"][:3])


sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

def process_func(examples):
    tokenized_examples = tokenizer(text=sample_dataset["question"],
                               text_pair=sample_dataset["context"],
                               return_offsets_mapping=True,
                               return_overflowing_tokens=True,
                               stride=128,
                               max_length=384, 
                               truncation="only_second",
                               padding="max_length")
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")   # 获取出overflow_to_sample_mapping的信息
    start_positions = []     # 记录answer在context中的起始位置
    end_positions = []       # 记录answer在context中的终止位置
    example_ids = []         # 记录数据的唯一id

    for idx, _ in enumerate(sample_mapping):
        answer = examples["answers"][sample_mapping[idx]]                   # 获取第idx个answers字段对应的值

        start_char = answer["answer_start"][0]                              # answer在context中起始位置的字符
        end_char = start_char + len(answer["text"][0])                      # answer在context中终止位置的字符
        
        context_start = tokenized_examples.sequence_ids(idx).index(1)                           # 获取 1 
        context_end = tokenized_examples.sequence_ids(idx).index(None, context_start) - 1       # 获None对应的

        offset = tokenized_examples.get("offset_mapping")[idx]

        if offset[context_end[1]] < start_char or offset[context_start][0] > end_char:
            start_token_pos = 0
            end_token_pos = 0
        else:
            token_id = context_start
            while token_id <= context_end and offset[token_id][0] < start_char:
                token_id += 1
            start_token_pos = token_id
            token_id = context_end
            while token_id >= context_start and offset[token_id] [1] > end_char:
                token_id -= 1
            end_token_pos = token_id
        start_positions.append(start_token_pos)
        end_positions.append(end_token_pos)
        example_ids.append(examples["id"][sample_mapping[idx]])

        # 设置多个滑动窗口区域中的answer的起始和终止位置信息
        # 即当预测处理的answer对应到context中mapping是Nond时, 代表当前滑动窗口区域中的context不存在answer的答案
        tokenized_examples["offset_mapping"][idx] = [
            (o if tokenized_examples.sequence_ids(idx)[k] == 1 else None)
            for k, o, in enumerate(tokenized_examples["offset_mapping"][idx])
        ]

    
    tokenized_examples["example_ids"] = example_ids
    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    return tokenized_examples

tokenized_datasets = tokenizer(datasets, truncation="only_first", max_length=128, padding="max_length")
           
# 获取模型的预测输出
# 即将模型对滑动窗口中的各个部分预测的结果进行聚合得到最终的结果
# 需要模型预测在起始位置的logits、结束位置的logits、原始数据集、mapping信息
def get_results(start_logits, end_logits, examples, features):
    predictions = {}
    references = {}
    example_to_feature = collections.defaultdict(list)
    for idx, example_id in enumerate(features["example_ids"]):
        example_to_feature[example_id].append(idx)
    

    # 最优答案的前20个
    n_best = 20
    # 最大答案的长度
    max_answer_length = 30

    for example in examples:
        example_id = example["idx"]
        context = example["context"]
        answers = []

        # 对每个example中每个feature进行遍历, 找到里面所有符合要求的答案
        for feature_idx in example_to_feature[example_id]:
            start_logit = start_logits[feature_idx]
            end_logit = end_logits[feature_idx]
            offset = features[feature_idx]["offset_mapping"]
            start_indexs = np.argsort(start_logit)[::-1][:n_best].tolist()
            end_indexs = np.argsort(end_logit)[::-1][:n_best].tolist()

            # 对每一个start_index和每一个end_index进行遍历
            for start_index in start_indexs:
                for end_index in end_indexs:
                    if offset[start_index] is None or offset[end_index] is None:        # 说明不是它不是answer在context中的位置, 说明不是答案的位置, 需要过滤
                        continue
                    if end_index < start_index  or end_index - start_index + 1 > max_answer_length:  # 如果超过最大长度, 也是的不合理的答案位置, 还是要过滤
                        continue
                    
                    # 构造answer的所有信息
                    answers.append(
                     {
                        "text": context[offset[start_index][0]: offset[end_index][1]],
                        "score": start_logit[start_index] + end_logit[end_index]
                     }
                    )

        # 如果取到了答案
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["score"])        # 获取分数最大对应的答案
            predictions[example_id] = best_answer["text"]
        else:   # 如果没有符合要求的答案
            predictions[example_id] = ""                            # 直接赋值为空

        references[example_id] = example["answers"]["text"]         # 标签


# 创建评估函数
def metric(pred):
    start_logits, end_logits = pred[0]
    if start_logits.shape[0] == len(tokenized_datasets["validatino"]): # 说明需要进行验证数据
        pred, references = get_results(start_logits, end_logits, datasets["validation"], tokenized_datasets["validation"])
    else:
        pred, references = get_results(start_logits, end_logits, datasets["test"], tokenized_datasets["test"])
    
    return evaluate_cmrc(pred, references)



# 加载模型
model = AutoModelForQuestionAnswering.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model_cache/chinese-macbert-base")


# 配置训练参数
training_args = TrainingArguments(
    output_dir="/Users/icur/CursorProjects/FineTuneBase/outputs/mrc",
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
    eval_dataset=tokenized_datasets["validation"],
    data_collator=DefaultDataCollator(),
    compute_metrics=metric
)



# 模型训练
trainer.train()


# 模型预测
pipe = pipeline()
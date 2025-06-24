"""
    本次使用截断方式处理过长context(上下文)的阅读理解任务

"""

import torch
import evaluate
from datasets import load_dataset, load_from_disk
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments, DefaultDataCollator


# 加载数据
datasets = load_from_disk("/Users/icur/CursorProjects/FineTuneBase/data/cmrc2018")
print(datasets["train"][0])

# 加载模型
# snapshot_download(
#     repo_id="hfl/chinese-macbert-base",  # 模型ID
#     local_dir="/Users/icur/CursorProjects/FineTuneBase/model_cache/chinese-macbert-base"
# )




# 数据预处理
tokenizer = AutoTokenizer.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model_cache/chinese-macbert-base")
print(tokenizer)


# 数据预处理MRC问题的最难点, 也是最核心的点
# 先使用部分数据来解释数据预处理的方法

sample_dataset = datasets["train"].select(range(10))

tokenized_examples = tokenizer(text=sample_dataset["question"], text_pair=sample_dataset["context"])
# print(tokenized_examples["input_ids"][0], len(tokenized_examples["input_ids"][0]))      # 可以看出光context上下文编码之后的长度就767, 超过了模型的最大输入长度



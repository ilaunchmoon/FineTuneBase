"""
    因果语言模型(Causal LM): AutoModelForCausalLM

    
    模型: bloom-389m-zh


    注意: CausalLM不需要输入labels标签


"""
from torch.utils.data import DataLoader
from datasets import Dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BloomModel, pipeline

# 下载模型
# snapshot_download(
#     repo_id="Langboat/bloom-389m-zh",  # 模型ID
#     local_dir="/Users/icur/CursorProjects/FineTuneBase/model_cache/Langboat"
# )


# 加载数据集
data_dir = "/Users/icur/CursorProjects/FineTuneBase/data/wiki_cn_filtered"
ds = Dataset.load_from_disk(data_dir)
print(ds)


# 数据预处理
model_dir = "/Users/icur/CursorProjects/FineTuneBase/model_cache/Langboat"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
# print(tokenizer)


def process_func(examples):
    return tokenizer(examples["completion"], max_length=384, truncation=True)

# input_ids、token_type_ids、attention_mask, 注意因果模型中不需要输入labels, 所以它对应tokenzier也没有这个字段
tokenized_ds = ds.map(process_func, batched=True, remove_columns=ds.column_names)
# print(tokenized_ds)


# 处理为批处理格式的DataLoader, 仅仅为了看看编码之后的结果是什么
# DataCollatorForLanguageModeling()中代码预训练模型的dataCollator, mlm=False代表为因果预训练模型
# input_ids、attention_mask是输入到llm中的字段
dl = DataLoader(tokenized_ds, batch_size=2, collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))
# print(next(enumerate(dl)))        # 打印第一个元素看看编码后的结果  其中<pad>编码为3, </s>结束token为2



# 创建模型
model = AutoModelForCausalLM.from_pretrained(model_dir)


# 配置训练参数
args = TrainingArguments(
    output_dir="/Users/icur/CursorProjects/FineTuneBase/outputs/causal_llm",
    per_device_train_batch_size=4, 
    gradient_accumulation_steps=8,
    logging_steps=10,
    num_train_epochs=1
)

# 创建训练器
trainer = Trainer(
    args=args,
    model=model,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)



# 开启训练
trainer.train()


# 开启测试
# text-generation代表生成模型
# 生成的最大长度 max_length, do_sample为True代表生成采样
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=128, do_sample=True)
response = pipe("在微调大型模型时, 对于不同层的参数可以设置不同的weight_decay值")
print(response)




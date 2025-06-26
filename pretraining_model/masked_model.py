"""
    掩码语言模型(Mask LM)

        AutoModelForMaskedLM


"""
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline


# 加载数据集
data_dir = "/Users/icur/CursorProjects/FineTuneBase/data/wiki_cn_filtered"
ds = Dataset.load_from_disk(data_dir)
print(ds)


# 数据预处理
model_dir = "/Users/icur/CursorProjects/FineTuneBase/model_cache/chinese-macbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_dir)

def process_func(examples):
    return tokenizer(examples["completion"], max_length=384, truncation=True)

# input_ids、token_type_ids、attention_mask、labels, labels为-100的数代表需要模型预测处理的token, 也就是使用特殊标记MASK遮掩掉的token, MASK会使用103来进行表示
tokenized_ds = ds.map(process_func, batched=True, remove_columns=ds.column_names)
# print(tokenized_ds)

# 处理为批处理格式的DataLoader, 仅仅为了看看编码之后的结果是什么
# DataCollatorForLanguageModeling()中代码预训练模型的dataCollator, mlm=True代表为掩码预训练模型, mlm_probability代表随机遮掩调的token比例, 0.15代表随机遮掩15%的token
# input_ids、token_type_ids、attention_mask是输入到llm中的字段
dl = DataLoader(tokenized_ds, batch_size=2, collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15))
# print(next(enumerate(dl)))        # 打印第一个元素看看编码后的结果



# 创建模型
model = AutoModelForMaskedLM.from_pretrained(model_dir)



# 配置训练参数
args = TrainingArguments(
    output_dir="/Users/icur/CursorProjects/FineTuneBase/outputs/mask_lm_pretrain",
    per_device_train_batch_size=32, 
    logging_steps=10,
    num_train_epochs=1
)

# 创建训练器
trainer = Trainer(
    args=args,
    model=model,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
)



# 开启训练
trainer.train()



# 开启测试
# fill-mask代表掩码模型
pipe = pipeline("fill-mask", model=model, tokenizer=tokenizer,)
response = pipe("在[MASK][MASK]大型模型时, 对于不同层的参[MASK]可以设置不同的weight_decay值, 例如, 底层参数（如 BERT 的[MASK][MASK]层）可以设置较小的值, 顶层参数可以[MASK][MASK]较大的值")
print(response)

"""
    使用生成式对话模型来演示BitFit微调方式

    BitFit微调思想: 选择模型参数中的bias部分进行训练, 其他参数都冻结住



"""

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments, pipeline


# 加载数据
data_dir = "/Users/icur/CursorProjects/FineTuneBase/data/alpaca_data_zh"
ds = Dataset.load_from_disk(dataset_path=data_dir)

# 数据分割
ds = ds.train_test_split(0.1, seed=42)


# 数据预处理
model_dir = "/Users/icur/CursorProjects/FineTuneBase/model_cache/Langboat"
tokenizer = AutoTokenizer.from_pretrained(model_dir)


# 注意当前预处理还没有将文本转为tensor格式的编码, 还是文本信息
def process_func(examples):
    max_len = 256
    input_ids, attention_mask, labels = [], [], []

    # 将instruction字段和input拼接到一起作为模型输入
    instruction = tokenizer("\n".join(["Human: "+ examples["instruction"] + examples["input"]]).strip() + "\n\n Assistant: ")

    # 将output字段和结束标注token结合在一起, 作为真实标签
    response = tokenizer(examples["output"] + tokenizer.eos_token)
    
    # 将上面编码结果中各自input_ids、attention_mask、labels字段各种组合添加到list中
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = -100 * len(instruction["labels"] + response["labels"])
    
    # 针对大于最大长度的进行阶段
    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        attention_mask = attention_mask[:max_len]
        labels = labels[:max_len]
        
    return  {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


tokenized_ds = ds.map(process_func, batched=True, remove_columns=ds["train"].column_names)


# 创建模型
model = AutoModelForCausalLM.from_pretrained(model_dir)


# 将模型中所有可训练参数的bias都冻结住, 不让它参与训练
# 这就是它最为核心的思想
num_param = 0
for name, param in model.named_parameters():
    if "bias" in name:
        param.requires_grad = False
    else:
        num_param += param.numel()


# 可训练参数比例
trainable_param_rate = num_param / sum(param.numel() for param in model.parameters())



# 配置训练参数
training_args = TrainingArguments(
    output_dir="/Users/icur/CursorProjects/FineTuneBase/outputs/gerenate_chatbot",
    per_device_train_batch_size=10,
    per_device_eval_batch_size=20,
    num_train_epochs=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
)

# 创建训练器
trainer = Trainer(
    args=training_args,
    model=model,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)

# 开启训练
trainer.train()


# 推理测试
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
inputs = "Human: {} \n {}".format("考试有哪些高分技巧?", "").strip() + "\n\n Assistant: "
response1 = pipe(inputs, max_length=256, do_sample=True)
response2 = pipe(inputs, max_length=256, top_k=10)
response3 = pipe(inputs, max_length=256, top_p=0.7)

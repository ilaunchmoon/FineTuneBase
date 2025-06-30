import torch 
import pandas as pd 
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments 
from peft import LoraConfig, get_peft_model, PeftModel, TaskType, DataCollatorForSeq2Seq


# 加载模型和tokenizer
model_dir = ""
model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)


# 数据预处理
def process_func1(example):
    MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token, 因此需要放开一些最大长度, 保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    
    # add_special_tokens 不在开头加 special_tokens
    # 因为<|begin_of_text|>这些就是特殊tokens
    # 这部分必须给system和user角色已经关于assistant特殊标记token进行tokenizer
    instruction = tokenizer(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n现在你要扮演皇帝身边的女人--甄嬛<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  

    # response是作为assistant角色的提示词, 也就是训练数据的真实标签, 并且必须给assistant添加结束标识符<|eot_id|>
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)

    # 注意其实tokenizer.pad_token_id起到了结束标识token作用, 所以最好使用[tokenizer.eos_token_id]来替换
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为<|eot_id|>咱们也是要关注的所以补充为1, 如果不关注结束标识符, 模型会无休止的续写

    # 关于标签仅能使用assistant的token来计算loss, 所以其他部分token都使用-100来遮掩
    # 注意其实tokenizer.pad_token_id起到了结束标识token作用, 所以最好使用[tokenizer.eos_token_id]来替换
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  

    # 做截断
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# 数据读取
data_dir = ""
df = pd.read_json(data_dir)
ds = Dataset.from_pandas(df)
tokenized_data = ds.map(process_func1, batched=True, remove_columns=ds.column_names)


# 配置LoraConfig()参数
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    r=8,
    lora_alpha=16,
    inference_mode=False,  # 训练模式
    lora_dropout=0.1
)


# 获取peft模型
peft_model = get_peft_model(model=model, peft_config=lora_config)
peft_model.print_trainable_parameters()


# 配置训练参数
training_args = TrainingArguments(
    output_dir="/Users/icur/CursorProjects/FineTuneBase/outputs/huan",
    per_device_train_batch_size=10,
    per_device_eval_batch_size=20,
    num_train_epochs=1,
    save_strategy="epoch",
    eval_strategy="epoch",
    logging_steps=100,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
)

# 创建训练器
trianer = Trainer(
    args=training_args,
    model=peft_model,
    train_dataset=tokenized_data,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
)


# 开启训练
trianer.train()


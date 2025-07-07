"""
    QLora思想

        本质就是在加载基座模型的时候对基座模型进行NF4bit的量化操作, 能够进一步降低显存资源
        在微调训练的时候量化的基座模型参数被冻结, 不会参与更新
        并且在微调训练过程中, 基座模型前向计算和方向计算都会将当前需要的基座模型参数进行反量化, 然后再进行前向传播或反向传播

        


        
    做法:

        step1: 配置BitsAndBytesConfig()用于QLora的量化参数

        step2: 使用上一步配置量化参数, 对基座模型进行4bit量化, 再进行加载, 即加载一个被量化的基座模型

        step3: 对已经量化的基座模型在微调训练中提供反量化、修复梯度检查点问题等功能  model = prepare_model_for_kbit_training(model)  # 传入的参数model是一个量化的基座模型

        step4: 使用LoarConfig配置Lora微调所需要的参数

        step5: 使用get_peft_model()获取Qlora微调模型

        step6: 配置训练参数和训练器, 开启微调训练

        注意: 以上这几步严格按照如上顺序, 尤其是不能先使用get_peft_model()获取Qlora微调模型, 再执行model = prepare_model_for_kbit_training(model), 否则一定会错

"""


from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_kbit_training
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    DataCollatorForSeq2Seq, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig  # 新增：量化配置
)
import torch

# 加载数据集
ds = Dataset.load_from_disk("/Users/icur/CursorProjects/FineTuneBase/data/alpaca_data_zh")

# 数据预处理函数保持不变
def process_func(examples):
    MAX_LENGTH = 256
    input_ids_list, attention_mask_list, labels_list = [], [], []
    
    for i in range(len(examples["instruction"])):
        instruction_text = examples["instruction"][i]
        input_text = examples["input"][i] if examples["input"][i] else ""
        
        instruction_text = "\n".join(["Human: " + instruction_text, input_text]).strip() + "\n\n Assistant: "
        
        instruction_encoded = tokenizer(instruction_text)
        response_encoded = tokenizer(examples["output"][i] + tokenizer.eos_token)
        
        input_ids = instruction_encoded["input_ids"] + response_encoded["input_ids"]
        attention_mask = instruction_encoded["attention_mask"] + response_encoded["attention_mask"]
        labels = [-100] * len(instruction_encoded["input_ids"]) + response_encoded["input_ids"]
        
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)
    
    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    }

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model_cache/Langboat")

# 处理数据集
tokenized_ds = ds.map(process_func, batched=True, remove_columns=ds.column_names)

# ========== QLoRA关键修改1：量化配置 ==========
# 其实这些参数也可以直接在.from_pretrained()中进行设置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # 启用4位量化加载
    bnb_4bit_quant_type="nf4",      # 使用NormalFloat4量化类型, 即4bit量化, 使用的是4bit的分位量化技术,
    bnb_4bit_compute_dtype=torch.bfloat16,  # 模型前向或反向计算时使用bfloat16, torch.bfloat16就是torch.half
    bnb_4bit_use_double_quant=True, # 启用双重量化进一步压缩, 即双重量化 量化常数
)

# ========== QLoRA关键修改2：量化加载模型 ==========
model = AutoModelForCausalLM.from_pretrained(
    "/Users/icur/CursorProjects/FineTuneBase/model_cache/Langboat",
    quantization_config=bnb_config,  # 应用量化配置
    device_map="auto",               # 自动分配设备
    torch_dtype=torch.bfloat16       # 模型数据类型
)

# ========== QLoRA关键修改3：准备模型用于k位训练 ==========
model = prepare_model_for_kbit_training(model)

# ========== QLoRA关键修改4：LoRA配置 ==========
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["query_key_value", "dense_4h_to_h"],
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    modules_to_save=["word_embeddings"],
    bias="none"  # 明确指定不使用偏置项
)

# 创建QLoRA模型
model = get_peft_model(model, config)

# 打印可训练参数
print(model.print_trainable_parameters())

# ========== QLoRA关键修改5：训练参数调整 ==========
training_args = TrainingArguments(
    output_dir="/Users/icur/CursorProjects/FineTuneBase/outputs/gerenate_chatbot",
    per_device_train_batch_size=8,   # 可能需要降低batch size以适应显存
    gradient_accumulation_steps=2,    # 使用梯度累积补偿batch size减小
    num_train_epochs=1,
    evaluation_strategy="no",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=2e-4,              # QLoRA通常使用稍高的学习率
    fp16=True,                       # 使用混合精度训练
    optim="paged_adamw_8bit",        # 使用分页的8位优化器, 注意这是Qlora特有的分页优化器, 还可以是: paged_adamw_8bit、paged_adamw_32bit, 可以从OptimizerNames()中查询
    report_to="none",
    max_grad_norm=0.3,               # 梯度裁剪
    warmup_ratio=0.03,               # 学习率预热
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

# ========== QLoRA关键修改6：保存与加载 ==========
# 保存适配器权重
model.save_pretrained("/Users/icur/CursorProjects/FineTuneBase/outputs/qlora_adapter")

# 加载适配器权重
peft_model = PeftModel.from_pretrained(
    model=model, 
    model_id="/Users/icur/CursorProjects/FineTuneBase/outputs/qlora_adapter",
    is_trainable=False
)

# 推理测试
input_text = "Human: {} \n {}".format("考试有哪些高分技巧?", "").strip() + "\n\n Assistant: "
ipts = tokenizer(input_text, return_tensors="pt").to(model.device)

# 生成响应
response = peft_model.generate(
    **ipts,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

# 解码输出
print(tokenizer.decode(response[0], skip_special_tokens=True))
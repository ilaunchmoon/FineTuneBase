"""
    hard-prompt-tuning 配置Prompt-Tuning的参数步骤

        from peft import PromptTuningConfig, get_peft_model, TaskType, PromptTuningInit, PeftModel

        PromptTuningConfig:  用于配置Prompt-Tuning微调的参数信息

        get_peft_model: 通过它结合PromptTuningConfig获取一个Prompt-Tuning微调的模型

        TaskType: 任务类型, peft有很多中任务类型, 所以需要它来指明任务类型

        PromptTuningInit: 用于配置到底是hard-prompt, 还是soft-prompt

        PeftModel: 用于加载微调训练好的模型


        step1: 配置prompt-tuning的微调参数config

               task_type: 任务类型
               prompt_tuning_init: 设定prompt的类型, 一般为PromptTuningInit.TEXT
               prompt_tuning_init_text: 由于是hard-prompt-tuning, 则需要用这个设定这个hard-prompt的具体内容
               num_virtual_tokens: 用于设定hard-prompt的长度, 一般会使用基座模型tokenizer直接将hard-prompt的具体内容进行词嵌入, 然后获取它对应词嵌入的input_ids的长度
               tokenizer_name_or_path: 由于是上面一步需要使用tokenizer对hard-prompt的具体内容进行词嵌入, 所以这样要设定基座模型词嵌入的存放路径


        step2: 使用get_peft_model(), 结合prompt-tuning配置好的参数config和基座模型创建微调的模型, 注意需要PromptTuningInit来进行hard-prompt初始化

        
        step3: 其他都和之前一样

        
        step4: 加载微调好的模型

                使用PeftModel(model, mode_id="微调训练好的检查点文件路径"), model为基座模型, 来加载微调好的模型





"""
from peft import PromptTuningConfig, get_peft_model, TaskType, PromptTuningInit, PeftModel
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, pipeline


# 加载数据集
ds = Dataset.load_from_disk("/Users/icur/CursorProjects/FineTuneBase/data/alpaca_data_zh")
# print(ds)


# 数据预处理
tokenizer = AutoTokenizer.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model_cache/Langboat")

def process_func(examples):
    MAX_LENGTH = 256
    input_ids_list, attention_mask_list, labels_list = [], [], []
    
    # 遍历批量中的每个样本
    for i in range(len(examples["instruction"])):
        # 获取当前样本的指令和输入
        instruction_text = examples["instruction"][i]
        input_text = examples["input"][i] if examples["input"][i] else ""
        
        # 构建完整的输入文本
        instruction_text = "\n".join(["Human: " + instruction_text, input_text]).strip() + "\n\n Assistant: "
        
        # 对指令和回复进行编码
        instruction_encoded = tokenizer(instruction_text)
        response_encoded = tokenizer(examples["output"][i] + tokenizer.eos_token)
        
        # 构建模型输入
        input_ids = instruction_encoded["input_ids"] + response_encoded["input_ids"]
        attention_mask = instruction_encoded["attention_mask"] + response_encoded["attention_mask"]
        labels = [-100] * len(instruction_encoded["input_ids"]) + response_encoded["input_ids"]
        
        # 截断序列到最大长度
        if len(input_ids) > MAX_LENGTH:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]
            labels = labels[:MAX_LENGTH]
        
        # 保存处理后的样本
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)
    
    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    }


# tokenized化数据
tokenized_ds = ds.map(process_func, batched=True, remove_columns=ds.column_names)



# 加载模型
model = AutoModelForCausalLM.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model_cache/Langboat")


# 配置prompt tuning
config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,                   # 指明为因果生成任务, TaskType类下有很多中任务: 如SEQ_CLS(文本分类)等
    prompt_tuning_init=PromptTuningInit.TEXT,       # 代表hard-prompt的初始化prompt是一段固定的文本内容, 还可以title, 随后需要使用prompt_tuning_init_text指定hard-prompt的具体内容
    prompt_tuning_init_text="下面是一段对话机器人对话", # 设置这段提示词的具体内容
    num_virtual_tokens=len(tokenizer("下面是一段对话机器人对话")["input_ids"]),  # 由于是hard-prompt-tuning, 则需要设定具体prompt的内容, 内容长度就是设定的具体prompt的对应的input_ids的长度
    tokenizer_name_or_path="/Users/icur/CursorProjects/FineTuneBase/model_cache/Langboat",   # 由于需要使用tokenizer对设定的prompt进行tokenize, 从而获取prompt长度, 所以需要使用设定tokenizer的路径
)


# 根据prompt-tuning配置的参数来创建peft模型
model = get_peft_model(model, config)         # 其中参数model为基座模型, config就是配置prompt-tuning微调的参数, 返回的模型是能够用于进行微调的模型peftmodel

# print(model.print_trainable_parameters())   # 可以输出使用prompt-tuning微调方法的可训练参数为多少: trainable params: 10,240 || all params: 345,779,200 || trainable%: 0.0030



# 其他都不变

# 配置训练参数
training_args = TrainingArguments(
    output_dir="/Users/icur/CursorProjects/FineTuneBase/outputs/gerenate_chatbot",
    per_device_train_batch_size=10,
    per_device_eval_batch_size=20,
    num_train_epochs=1,
    eval_strategy="no",             # 没有设置eval_dataset, 关闭评估策略
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



# 加载微调训练好的模型
peft_model_checkpoint_dir=""

# model为基座模型, mode_id为微调好后存放的checkpoint文件的路径名
# 需要注意的是: 使用peft_model要确保输入的测试数据和微调后模型在同一个device上
peft_model = PeftModel.from_pretrained(model=model, mode_id=peft_model_checkpoint_dir)      



# 微调模型预测
# peft_model = peft_model.cuda() 用于转到cuda

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
input_text = "Human: {} \n {}".format("考试有哪些高分技巧?", "").strip() + "\n\n Assistant: "
inputs = tokenizer(input_text, return_tensors="pt").to(peft_model.device)
print(tokenizer.decode(peft_model.generate(**inputs, max_length=128, do_sample=True)[0], skip_special_tokens=True))



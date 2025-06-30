"""
    lora 配置 lora的参数步骤

        from peft import LoraConfig, TaskType, get_peft_model, 

        LoraConfig:  用于配置lora微调的参数信息

        get_peft_model: 通过它结合LoraConfig获取一个lora微调的模型

        TaskType: 任务类型, peft有很多中任务类型, 所以需要它来指明任务类型

        


    lora 配置

        step1: 配置lora的微调参数config
                
            task_type=TaskType.CAUSAL_LM,                                               # 指明为因果生成任务, TaskType类下有很多中任务: 如SEQ_CLS(文本分类)等
            target_modules=["query_key_value", "dense_4h_to_h"],                        # 指定LLM中到底哪些模块的参数参与lora微调训练, 还可以通过模型参数名称结合正则匹配的方法来匹配到某一些模块的参数来进行微调, 如".*\.1.*query_key_value"代表层数以1开头的query_key_value模块参与lora微调
            r=8,                                                                        # 指定lora微调的低秩矩阵的秩为8, 一般都设置为8或16, 少数情况下设置为32
            lora_alpha=16,                                                              # 指定lora微调的低秩矩阵的缩放因子, 一般为r的2倍
            lora_dropout=0.1,                                                           # 指定LLM中到底哪些模块的参数参与lora微调训练中的dropout_rate
            modules_to_save=["word_embedings"],                                         # 除了target_modules中指定的微调参数模块, 还可以通过modules_to_save来指定某个其他参数模块进行微调, 比如分类模型中的分类头就是一个MLP, 通过modules_to_save可以额外设置任务头的参数参与微调训练

                
            注意: 这些都会影响最终可训练的参数量

        step2: 使用get_peft_model(), 结合lora配置好的参数config和基座模型创建微调的模型

        
        step3: 其他都和之前一样

                训练参数配置、测试参数配置都一样

        
        step4: 加载微调好的模型

                使用 peft_model = PeftModel(model, mode_id="微调训练好的检查点文件路径"), model为基座模型, 来加载微调好的模型


        step5: 将lora微调好的检查点文件和基座模型进行合并

                使用上一步的加载微调好的模型, 进行直接合并

                merge_model = peft_model.merge_and_unload()               # 不将合并模型存放在本地

                merge_model.save_pretrained("本地路径")                    # 将合并好的模型存放在本地









"""
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer


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

# 获取该模型中所有参数, 方便lora微调target_modules参数指定到底哪些模型参数参与微调训练
# lora微调的模型参数主要集中在注意力模块、FNN模块、还有就是某些分类模型中的分类头模块
for name, param in model.parameters():
    print(name)     


# 配置prompt tuning
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,                                               # 指明为因果生成任务, TaskType类下有很多中任务: 如SEQ_CLS(文本分类)等
    target_modules=["query_key_value", "dense_4h_to_h"],                        # 指定LLM中到底哪些模块的参数参与lora微调训练, 还可以通过模型参数名称结合正则匹配的方法来匹配到某一些模块的参数来进行微调, 如".*\.1.*query_key_value"代表层数以1开头的query_key_value模块参与lora微调
    r=8,                                                                        # 指定lora微调的低秩矩阵的秩为8
    lora_alpha=16,                                                              # 指定lora微调的低秩矩阵的缩放因子, 一般为r的2倍
    lora_dropout=0.1,                                                           # 指定LLM中到底哪些模块的参数参与lora微调训练中的dropout_rate
    modules_to_save=["word_embedings"],                                         # 除了target_modules中指定的微调参数模块, 还可以通过modules_to_save来指定某个其他参数模块进行微调, 比如分类模型中的分类头就是一个MLP, 通过modules_to_save可以额外设置任务头的参数参与微调训练                                          

)



# 根据lora配置的参数来创建peft模型
model = get_peft_model(model, config)         # 其中model为基座模型, config就是配置lora微调的参数


# print(model.print_trainable_parameters())   # 可以输出使用lora微调方法的可训练参数为多少


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
peft_model = PeftModel.from_pretrained(model=model, model_id=peft_model_checkpoint_dir)   

# 将基座模型和lora微调训练得到权重进行合并
merged_model = peft_model.merge_and_unload()


# 将合并模型保存到本地
merged_model_dir = ""
merged_model.save_pretrained(merged_model_dir)

# 推理测试
input_text = "Human: {} \n {}".format("考试有哪些高分技巧?", "").strip() + "\n\n Assistant: "
ipts = tokenizer(input_text, return_tensors="pt")

# 使用peft_model进行推理微调后模型的结果
response = tokenizer.decode(peft_model.generatr(**ipts, do_sample=True)[0], skip_special_tokens=True)
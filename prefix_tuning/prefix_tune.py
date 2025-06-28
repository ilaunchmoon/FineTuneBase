"""
    Prefix-tuning 配置 Prefix-tuning的参数步骤

        from peft import PrefixTuningConfig, TaskType, get_peft_model, PeftModel

        PrefixTuningConfig:  用于配置Prefix-tuning微调的参数信息

        get_peft_model: 通过它结合PromptEncoderConfig获取一个Prefix-tuning微调的模型

        TaskType: 任务类型, peft有很多中任务类型, 所以需要它来指明任务类型

        PeftModel: 用于加载基座模型和Prefix-tuning微调训练好的检查点文件, 即类似将基座模型和微调好的权重合并成一个模型, 这个模型就是微调训练好的模型



    Prefix-tuning 操作步骤

        step1: 初始化可训练前缀向量
        step2: 投影层转换​​(可选), 将前缀prompt词嵌入结果进行MLP投影编码, 它是通过prefix_projection来设置是否开启, encoder_hidden_size来设置这个MLP的隐藏层维度
        step3: 分层参数化 & Key/Value 投影​​
        step4: 整合进 past_key_values
        


    Prefix-tuning 配置

        step1: 配置Prefix-tuning的微调参数config
                
                task_type=TaskType.CAUSAL_LM,           # 指明为因果生成任务, TaskType类下有很多中任务: 如SEQ_CLS(文本分类)等
                prefix_projection=True,                 # 用于设置是否使用的全连接层(默认为False), 即将prefix-prompt词嵌入的结果是否先经过全连接层, 一般设置为True会引入更多的可学习参数, 能够让微调效果更好
                encoder_hidden_size=2048,               # 设置编码层的隐藏维度, 即将prompt词嵌入结果进行编码的编码层的隐藏层维度, 注意它不是LLM注意力模块的编码器, 而是进入注意力机制前的一个编码层
                num_virtual_tokens=10,                  # 由于是soft-prompt-tuning, 则只需要设定prompt的长度, 不需要设定具体prompt的内容, 如果是hard-prompt-tuning的话还需要设置具体prompt内容

                
                注意: 这些都会影响最终可训练的参数量

        step2: 使用get_peft_model(), 结合prompt-tuning配置好的参数config和基座模型创建微调的模型

        step3: 其他都和之前一样

                训练参数配置、测试参数配置都一样

        
        step4: 加载微调好的模型

                使用PeftModel(model, mode_id="微调训练好的检查点文件路径"), model为基座模型, 来加载微调好的模型







"""
from peft import PrefixTuningConfig, TaskType, get_peft_model, PeftModel
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


# 配置prompt tuning
config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,           # 指明为因果生成任务, TaskType类下有很多中任务: 如SEQ_CLS(文本分类)等
    prefix_projection=True,                 # 用于设置是否使用的全连接层(默认为False), 即将prefix-prompt词嵌入的结果是否先经过全连接层
    encoder_hidden_size=2048,               # 设置编码层的隐藏维度, 即将prompt词嵌入结果进行编码的编码层的隐藏层维度
    num_virtual_tokens=10,                  
)



# 根据Prefix-tuning配置的参数来创建peft模型
model = get_peft_model(model, config)         # 其中model为基座模型, config就是配置Prefix-tuning微调的参数

# print(model.print_trainable_parameters())   # 可以输出使用Prefix-tuning微调方法的可训练参数为多少


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
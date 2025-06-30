"""
        adalora 配置 adalora的参数步骤

        from peft import adaloraConfig, TaskType, get_peft_model, 

        adaloraConfig:  用于配置adalora微调的参数信息

        get_peft_model: 通过它结合adaloraConfig获取一个adalora微调的模型

        TaskType: 任务类型, peft有很多中任务类型, 所以需要它来指明任务类型

        


    adalora 配置

        step1: 配置adalora的微调参数config
            
                task_type=TaskType.CAUSAL_LM,                           # 下游任务类型
                target_modules=["query_key_value", "dense_4h_to_h"],    # 针对特定模型的模块
                init_r=12,                                              # 初始秩
                target_r=8,                                             # 目标秩 ( 应小于初始秩 ) 
                beta1=0.85,                                             # 重要性度量的平滑系数beta1
                beta2=0.85,                                             # 重要性度量的平滑系数beta2
                tinit=200,                                              # 初始训练步数 ( 不进行秩修剪 ) 
                tfinal=1000,                                            # 达到目标秩的训练步数
                deltaT=10,                                              # 每10步进行一次秩调整
                adalora_alpha=16,                                       # adalora α值 ( 缩放因子 ) 
                adalora_dropout=0.1,                                    # adalora层Dropout率
                modules_to_save=["word_embeddings"],                    # 除adalora外额外训练的词嵌入层
                bias="none",                                            # 不使用偏置项
                # orth_reg_weight=0.5                                   # 用于loss函数中惩罚P和Q不为正交矩阵时的系数, 默认为0.5   
            
                
            注意: 这些都会影响最终可训练的参数量

        step2: 使用get_peft_model(), 结合adalora配置好的参数config和基座模型创建微调的模型

        
        step3: 其他都和之前一样

                训练参数配置、测试参数配置都一样

        
        step4: 加载微调好的模型

                使用 peft_model = PeftModel(model, mode_id="微调训练好的检查点文件路径"), model为基座模型, 来加载微调好的模型


        step5: 将adalora微调好的检查点文件和基座模型进行合并

                使用上一步的加载微调好的模型, 进行直接合并

                merge_model = peft_model.merge_and_unload()               # 不将合并模型存放在本地, 仅仅是将基座模型和微调训练的检查点文件进行合并

                merge_model.save_pretrained("本地路径")                    # 将合并好的模型存放在本地

        
        关于AdaLoraConfig()参数说明

            task_type ( TaskType, 可选):
                ​​功能​​: 指定任务类型 
                ​​目的​​: 帮助PEFT库针对特定任务 ( 如因果语言建模、序列分类等 ) 正确配置模型 
                ​​设置​​: 例如: TaskType.CAUSAL_LM、TaskType.SEQ_CLS等 根据你的任务选择合适的TaskType枚举值 

            target_modules ( Union[List[str], str], 必须设置):
                ​​功能​​: 要应用AdaLora的模块名称 
                ​​目的​​: 指定在模型的哪些模块上添加AdaLora层 通常这些模块是线性层 
                ​​设置​​: 可以是模块名的字符串列表, 如: ["q_proj", "v_proj"] 也可以使用正则表达式字符串 如果不知道, 可以设置为None, 但这样可能会对所有线性层应用, 导致参数过多 

            init_r ( int, 默认为12):
                ​​功能​​: 初始的秩 ( rank ) 值 
                ​​目的​​: 在训练开始前, 每个AdaLora层的初始秩 通常设置得比目标秩大一些, 以便在训练过程中有修剪空间 
                ​​设置​​: 可以设为12, 根据模型大小调整, 大模型可以更大一些 ( 如16 ) , 小模型可以小一些 ( 如6 )  

                小模型 (100M-1B): 4-8
                中等模型 (1B-7B): 8-12
                大模型 (>7B): 12-16
                作为动态调整的起点，较高的初始秩提供更大的表达容量


            target_r ( int, 默认为8):
                ​​功能​​: 最终的目标秩 
                ​​目的​​: 训练结束后, 每个AdaLora层将达到的目标秩 ( 实际上由于动态调整, 各层最终秩可能略有不同 )  
                ​​设置​​: 通常小于init_r, 例如8 可以根据需要调整, 比如为了更低的参数, 可以设置更小的值 ( 如4 )  
                建议设置总是小于init_r的值-通常为init_r的2/3


            beta1 ( float, 默认为0.85):
                ​​功能​​: 用于控制重要性分数累积的衰减率 
                ​​目的​​: 在更新重要性分数的指数移动平均 ( EMA ) 时, 作为平滑参数 ( 类似于Adam的beta1 )  它影响重要性估计的稳定性 
                ​​设置​​: 通常设置为0.85, 也可以尝试0.8到0.9之间的值 
                使用指数移动平均平滑梯度信息, beta1越大表示越依赖最近的梯度信息


            beta2 ( float, 默认为0.85):
                ​​功能​​: 与beta1类似, 但用于控制重要性分数平方的累积 
                ​​目的​​: 也是用于EMA, 对梯度信息进行平滑 
                ​​设置​​: 通常和beta1相同, 设置为0.85 
                与beta1共同作用, beta2值越大表示重要性度量变化越平滑


            tinit ( int, 默认为0):
                ​功能​​: 无修剪 ( no pruning ) 的初始训练步数 
                ​​目的​​: 在训练初期, 不进行秩的修剪, 允许模型先学习一段时间, 以便更准确地评估参数的重要性 
                ​​设置​​: 例如200, 表示前200步不修剪 
                小模型: 50-200步, 大模型: 200-500步, 前期不进行任何修剪，确保基本特征学习稳定


            tfinal ( int, 默认为0):
                ​​功能​​: 达到目标秩的时间步 ( 训练步数 )  
                ​​目的​​: 在tfinal步时, 模型的秩将下降到target_r 之后, 秩保持固定 
                ​​设置​​: 设置为一个较大的值, 如1000 ( 根据总训练步数来定 )  如果训练步数很多, 可以设置为总步数的一半或三分之二 
                设为目标总训练步数的60-80%, 线性计划从init_r减到target_r的时间框架

            deltaT ( int, 默认为1):
                ​​功能​​: 进行修剪的时间间隔 ( 步数 )  
                ​​目的​​: 每隔deltaT步, 对模型各层的秩进行一次调整 ( 修剪 )  
                ​​设置​​: 例如10, 表示每10步进行一次修剪 不宜过小 ( 计算开销大 ) , 也不宜过大 ( 调整不频繁 )  通常在10到50之间 
                10-50 (太频繁增加计算开销，太稀疏降低调整精度)
                每deltaT步进行一次秩调整(减少低重要性参数)


            lora_alpha ( int, 默认为None):
                ​​功能​​: LoRA层中的缩放因子 
                ​​目的​​: 在应用LoRA时, 将低秩矩阵的输出缩放 ( 除以rank后乘以lora_alpha ) , 相当于控制学习率大小 
                ​​设置​​: 如果为None, 则使用默认值 ( 通常为r的2倍 )  一般情况下设置为8、16或32等 可以视为与学习率有关的一个超参 



            lora_dropout ( float, 默认为0.0):
                ​​功能​​: LoRA层的dropout率 
                ​​目的​​: 防止过拟合, 随机丢弃一部分神经元 
                ​​设置​​: 0到1之间, 比如0.1 可以不用设置 ( 0 ) 或者小值 



            modules_to_save ( List[str], 可选):
                ​​功能​​: 除了LoRA层外, 还需要训练 ( 保存 ) 的模块 
                ​​目的​​: 对于某些模块 ( 如词嵌入层或输出层 ) , 有时需要完全微调, 这些模块将被指定并训练 
                ​​设置​​: 例如: ["word_embeddings"] 如果不需要额外训练其他模块, 则为None 



            bias ( str, 默认为"none"):
                ​​功能​​: 是否训练偏置项 
                ​​目的​​: 控制是否在训练中更新偏置参数 
                ​​设置​​: 可选值有"none" ( 不训练偏置 ) 、"all" ( 训练所有偏置 ) 或"lora_only" ( 仅训练LoRA层的偏置 )  



            layers_to_transform ( Union[List[int], int], 可选):
                ​​功能​​: 指定要转换的层的索引 
                ​​目的​​: 仅对指定层应用AdaLora 如果为None, 则适用于所有层 
                ​​设置​​: 例如: [0, 1, 2] 或 0 ( 仅一层 )  如果为None, 则所有层都被转换 


            layers_pattern ( str, 可选):
                ​​功能​​: 用于匹配要转换层的模式 ( 正则表达式 )  
                ​​目的​​: 与target_modules一起, 用于识别模型中哪些层要添加AdaLora 
                ​​设置​​: 通常不需要指定, 除非有特殊模块结构 



            rank_pattern ( dict, 默认为{}):
                ​​功能​​: 指定不同层的秩模式 ( 即不同层可以有不同的秩 )  
                ​​目的​​: 允许手动设置某些层具有特定的秩 ( 而非统一设置 )  
                ​​设置​​: 例如: {"attention.": 8, "mlp.": 4} 默认空字典表示所有层使用相同的秩 ( init_r和target_r )  


            alora_layers ( dict, 可选):
                功能​​: 内部使用, 通常不需要设置 



            inference_mode ( bool, 默认为False):
                ​​功能​​: 是否在推理模式下 
                ​​目的​​: 如果为True, 则加载的模型将用于推理 ( 不参与训练 )  通常在加载微调后的模型时使用 
                ​​设置​​: 训练时为False ( 默认 )  



"""


from peft import AdaLoraConfig, TaskType, get_peft_model, PeftModel
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    DataCollatorForSeq2Seq, 
    TrainingArguments, 
    Trainer
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

# ========== 非量化模型加载 ==========
model = AutoModelForCausalLM.from_pretrained(
    "/Users/icur/CursorProjects/FineTuneBase/model_cache/Langboat",
    device_map="auto",
    torch_dtype=torch.bfloat16  # 保持bfloat16精度, 提升训练速度
)

# ========== adalora配置 ==========
config = AdaLoraConfig(     
    task_type=TaskType.CAUSAL_LM,      # 下游任务类型
    target_modules=["query_key_value", "dense_4h_to_h"],  # 针对特定模型的模块
    init_r=12,                   # 初始秩
    target_r=8,                  # 目标秩 ( 应小于初始秩 ) 
    beta1=0.85,                  # 重要性度量的平滑系数beta1
    beta2=0.85,                  # 重要性度量的平滑系数beta2
    tinit=200,                   # 初始训练步数 ( 不进行秩修剪 ) 
    tfinal=1000,                 # 达到目标秩的训练步数
    deltaT=10,                   # 每10步进行一次秩调整
    adalora_alpha=16,               # adalora α值 ( 缩放因子 ) 
    adalora_dropout=0.1,            # adalora层Dropout率
    modules_to_save=["word_embeddings"],  # 除adalora外额外训练的词嵌入层
    bias="none",                   # 不使用偏置项
    # orth_reg_weight=0.5          # 用于loss函数中惩罚P和Q不为正交矩阵时的系数, 默认为0.5           
)

# 创建Adaadalora模型
model = get_peft_model(model, config)

# 打印可训练参数
print(model.print_trainable_parameters())

# ========== 训练参数 ==========
training_args = TrainingArguments(
    output_dir="/Users/icur/CursorProjects/FineTuneBase/outputs/adaadalora_chatbot",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=2,           # Adaadalora需要更多训练轮次 ( 建议2-3轮 ) 
    evaluation_strategy="no",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=2e-4,
    fp16=True,                    # 保持混合精度训练
    optim="adafactor",            # Adaadalora推荐使用Adafactor优化器
    report_to="none",
    max_grad_norm=0.3,            # 梯度裁剪
    warmup_ratio=0.03,            # 学习率预热
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

# ========== 保存与加载 ==========
# 保存适配器权重
model.save_pretrained("/Users/icur/CursorProjects/FineTuneBase/outputs/adaadalora_adapter")

# 加载适配器权重 ( 注意: 需要先加载原模型 ) 
base_model = AutoModelForCausalLM.from_pretrained(
    "/Users/icur/CursorProjects/FineTuneBase/model_cache/Langboat",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
peft_model = PeftModel.from_pretrained(
    model=base_model, 
    model_id="/Users/icur/CursorProjects/FineTuneBase/outputs/adaadalora_adapter"
).eval()

# 推理测试
input_text = "Human: {} \n {}".format("考试有哪些高分技巧?", "").strip() + "\n\n Assistant: "
ipts = tokenizer(input_text, return_tensors="pt").to(peft_model.device)

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


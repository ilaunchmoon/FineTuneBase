"""
    本节重点说明针对一个新发布的模型, 如何将训练数据处理为符合该模型的输入要求
    顺带使用半精度训练ChatGLM

    
    任何一个模型进行微调核心是解决数据处理和让数据能够输入到基座模型

        第一步: 先使用半精度加载模型来进行推理, 以得到该新模型的输入要求, 即做模型输入的数据的对齐

              tokenizer = AutoTokenizer.from_pretrained("")
              model = AutoModel.from_pretrained("", trust_remote_code=True, low_cpu_mem_uage=True, torch_dtype=torch.half, device_map="auto")

              # 直接测试模型输入, 看看它输出的内容要求那些字段、内容等
              model.chat(tokenizer, "你好, 有什么赚钱的方法", history=[])  


              # 一般会涉及对chat()方法的详细实现来判断输入到底是什么, 可以使用如下办法来获取任何一个函数的具体实现

              import inspect

              print(inspect.getsource(函数名))    # 获取函数源代码


        第二步: 通过分析输入数据之后的处理逻辑来得到输入数据的对齐, 即得到符合输入模型的数据格式

               最主要的办法就是查看model.chat()方法对提示词输入的格式来获取正确输入模型中的数据格式
    
               
    ChatGLM3模型训练注意事项

        1. 由于该模型支持FunctionCalling, 则在tokenizer处理输入提示词时, 必须给输入序列开头添加 "\n"

        2. 使用lora微调的时候, 如果不指定task_type的参数, 则需要在TrainingArguments参数中设置参数 remove_unused_columns=False, 否则训练会报错

        3. 使用model.chat()中的源码来获取指令模版和输入数据的格式进行对齐, 确保输入模型中的数据能够格式对齐

        

"""
import torch 
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments


# 加载数据
data_dir=""
data = Dataset.load_from_disk(data_dir)


# 数据预处理
model_dir=""
tokenzier = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)


# 定义数据预处理函数
def process_func(examples):
    MAX_LENGTH=356
    input_ids, attention_mask, labels = [], [], []
    instruction = "\n".join(examples["instruction"], examples["input"]).strip()         # 等价于模型要求输入的query
    # 当前模型使用build_chat_input()函数处理输入, 即能得到能够输入模型中的数据格式 [gMASK]sop<|user|> \n query<|assistant|>
    # [gMASK]sop<|user|> \n query<|assistant|> 可以看到在query都添加了一个 "\n"
    instruction = tokenzier.build_chat_input(instruction, history=[], role="user")      # 该模型的tokenizer编码结果都是以tensor形式返回

    # 注意不要直接将eos_token添加到example["output"]后面
    # 否则它会将它作为普通token来进行分词和对待
    # 因将它放在解码后的input_ids后面
    response = tokenzier("\n" + examples["output"], add_special_tokens=False)                                     # 该模型的tokenizer编码结果都是以tensor形式返回

    input_ids = instruction["input_ids"][0].numpy().tolist() + response["input_ids"] + [tokenzier.eos_token_id]   # 那么这里要与list直接相加需要将前面的转为list格式

    attention_mask = instruction["attention_mask"][0].numpy().tolist() + response["attention_mask"] + [1]          # 那么这里要与list直接相加需要将前面的转为list格式

    labels = [-100] * len(instruction["input_ids"][0].numpy().tolist()) + response["input_ids"] + [tokenzier.eos_token_id]

    return {
        "input_ids": input_ids[:MAX_LENGTH],
        "attention_mask": attention_mask[:MAX_LENGTH],
        "labels": labels[:MAX_LENGTH]
    }
    
# 获取数据预处理的数据
tokenized_ds = data.map(process_func, remove_columns=data.column_names)


# 加载模型
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.half, device="auto")


# 配置lora参数
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    target_modules=["q_proj"]
)

# 获取peft_model模型
model = get_peft_model(model=model, peft_config=lora_config)


# 输出可学习参数信息
model.print_trainable_parameters()



# 配置训练参数
output_dir=""
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2, 
    logging_steps=50,
    num_train_epochs=1,
    learning_rate=1e-5,
    save_strategy="epoch",
    eval_strategy="epoch",

)


# 创建训练器
trainer = Trainer(
    args=training_args,
    train_dataset=tokenized_ds.select(range(1000)),         # 选择前一千条来进行演示
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenzier, padding=True)
     
)



# 开启训练
trainer.train()



# 开始推理
model.eval()
print(model.chat(tokenzier, "考试有什么技巧", history=[], role="user")[0])



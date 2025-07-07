"""
    使用半精度训练llama2-7B需要注意如下几个问题

        0. 基座模型加载的时候必须使用半精度加载
           使用的是torch_dtype=torch.half参数来进行半精度加载
            
            model = AutoModelForCausalLM.from_pretrained(model_dir, low_cpu_mem_usage=True, trust_remote_code=True, device_map="auto", torch_dtype=torch.half)


        1. llama系列对中文语料支持不是很友好, 它的分词器可能会将一个中文字分词若个token
           为了能让模型能够尽可能地输入完整的数据(即不被提前截断), 应该将tokenizer的max_len设置大一些

        
        2. llama2系列默认是在输入序列的左侧来添加padding, 如果是这样进行训练, 模型无法收敛
           为了能够让模型微调训练的时候正常收敛, 必须将它的tokenizer添加padding方式修改为 padding_side="right" 从右侧添加padding

        
        3. 如果使用lora微调训练参数配置开启梯度检查点gradient_checkpointing=True
           此时必须使用model.enable_input_require_grads()对添加lora_linear替换的的peft_model支持: model.enable_input_require_grads()
           否则会报错, 因为基座模型梯度是被冻结的, 但是又要开启梯度简单, 所以必须对lora微调模型进行开启enable_input_require_grads()

        
        4. 如果需要对peft_model的lora模型中的lora替换的参数也做半精度处理, 就需要对peft_model, 执行 model.half()操作
           这样lora_linear层也会变成fp16, 否则lora_linear的参数默认就是单精度FP32
           并且如果将lora_linear也开启了半精度, 必须在训练器中配置优化器将adam_epsilon参数调大一些, 默认是1e-8
           注意: 如果发现训练的loss不正常, 而且此时使用模型训练的结果去推理, 发现model输出的logits中有很多NaN, 此时说明因为低精度的原因导致了溢出
                那么此时都可以考虑优化器的该参数调大一些

        
        5. llama2模型如果在tokenizer直接将 输入序列部分 + tokenizer.eos_token 的方法进行特殊结束token 来相加的方式
           tokenizer会把特殊结束标记eos_token当作一个正常token来进行编码, 从而导致模型无法学习到到底在哪结束生成过程
           即:

                tokenizer("abc"+ tokenizer.eos_token) 它会将tokenizer.eos_token当作一个正常的非结束特殊token进行编码


           为了能够让模型的tokenizer不将特殊的结束token也当作正常token进行编/解码, 有如下两种办法:

             第一: 给输入序列中添加特殊结束token, eos_token时, 在它的前面添加一个空格即可

                    tokenizer("abc "+ tokenizer.eos_token)          # 添加了一个token

             第二: 在tokenizer数据嵌入的时候, 直接在输入序列编码完后添加特殊结束token, 对应attention_mask需要添加 [1], labels最后也需要添加 eos_token

            
            注意: 如果设置好eos_token的问题, 还需要一个问题要修正: 在进行多批次训练时, 需要将pad_token_id设置为2, 否则训练之后模型的输出logits会出现NaN
                 tokenizer.pad_token_ids = 2

                






"""


import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments, pipeline


# 加载数据
data_dir = "/Users/icur/CursorProjects/FineTuneBase/data/alpaca_data_zh"
ds = Dataset.load_from_disk(dataset_path=data_dir)

# 数据分割
ds = ds.train_test_split(0.1, seed=42)


# 数据预处理
model_dir = "/Users/icur/CursorProjects/LowPrecision/model_cache/modelscope/Llama-2-7b-ms"

# 特别注意: 由于llama系列它默认是对输入序列左侧进行padding的, 如果是这样会造成错误
# 所以需要将它设置为右侧padding
tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="right")



# 注意当前预处理还没有将文本转为tensor格式的编码, 还是文本信息
def process_func(examples):
    max_len = 368               # 注意llama系列的模型对中文支持不是很友好, 所以最好将 max_len 设置比正常大一些, 否则很可能给重要信息截断掉
    input_ids, attention_mask, labels = [], [], []

    # 将instruction字段和input拼接到一起作为模型输入, 注意由于llama会给每句话都添加一个开始特殊token, 其实只需使用结束特殊标签eos即可, 为了统一先都不添加特殊token
    instruction = tokenizer("\n".join(["Human: "+ examples["instruction"], examples["input"]]).strip() + "\n\n Assistant: ", add_special_tokens=False)      

    # 将output字段和结束标注token结合在一起, 作为真实标签
    # llama2模型不能将output字段和结束标注token结合在一起, 否则它会将特殊结束token当作正常token进行编/解码
    # 应该在完成编码之后, 在input_ids最后添加, 对应的attention_mask也是在最后添加 + [1] 来代表特殊结束标签 eos_token 也需要被关注到, 这样模型就学会了在什么时候应该结束一句话的生成
    response = tokenizer(examples["output"],  add_special_tokens=False)
    
    # 将上面编码结果中各自input_ids、attention_mask、labels字段各种组合添加到list中
    # 并且在解码后input_ids添加结束token对于的token_id, attention_mask最后也需要添加[1] 代表最后的eos_token也需要被关注到
    # labels一样需要添加 eos_token
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]   
    labels = -100 * len(instruction["input_ids"] + response["input_ids"])  + [tokenizer.eos_token]
    
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
# 设置使用半精度加载, torch_dtype=torch.half
model = AutoModelForCausalLM.from_pretrained(model_dir, low_cpu_mem_usage=True, trust_remote_code=True, device_map="auto", torch_dtype=torch.half)

# lora微调训练时, 如果需要开启gradient_checkpointing=True来进行一步降低显存使用
# 必须先执行如下方法来对peft_model开启梯度检查, 否则此时基座模型是被冻结的, 无法开启gradient_checkpointing=True
model.enable_input_require_grads()

# 如果要让替换的lora矩阵 BA也是半精度, 那么需要对peft_model再次开启半精度
# 但是这个会有很大风险, 不建议这么做
# 如果非要这么做, 必须对优化器的epsion重新更大一些
model = model.half()




# 配置训练参数
training_args = TrainingArguments(
    output_dir="/Users/icur/CursorProjects/FineTuneBase/outputs/gerenate_chatbot",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=8,
    gradient_checkpointing=True,            # 开启梯度检查点, 需要保证模型加载之后使用 model.enable_input_require_grads(), 否则在lora微调的时候直接开启这个开关会报错
    # adam的优化器该是代表有效值范围, 默认是1e-8, 也就是说如果值小于1e-8, 则它会认为溢出, 此时会变为0, 所以需要将它的范围放大一些, 一般还是结合针对lora的矩阵BA也开启半精度训练时使用
    adam_epsilon=1e-4,   
    num_train_epochs=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
)

# 创建训练器
trainer = Trainer(
    args=training_args,
    model=model,
    train_dataset=tokenized_ds.select(range(3000)),         # 由于计算资源有限, 先选择前3000的数据来进行测试
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),                                    
)

# 开启训练
trainer.train()


# 推理测试
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
input_text = "Human: {} \n {}".format("考试有哪些高分技巧?", "").strip() + "\n\n Assistant: "
ipts = tokenizer(input_text, return_tensors="pt")
response = tokenizer.decode(model.generatr(**ipts, do_sample=True)[0], skip_special_tokens=True)
response1 = pipe(input_text, max_length=256, do_sample=True)
response2 = pipe(input_text, max_length=256, top_k=10)
response3 = pipe(input_text, max_length=256, top_p=0.7)




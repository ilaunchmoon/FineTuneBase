"""
    文本摘要任务


    模型: ChatGLM


    数据预处理: 


"""
import torch 
from rouge_chinese import Rouge
from datasets import Dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, pipeline

# 下载模型
model = AutoModelForSeq2SeqLM.from_pretrained("THUM/glm-large-chinese", model_dir="/Users/icur/CursorProjects/FineTuneBase/model_cache/glm", trust_remove_code=True)


# 加载数据
dataset_path = "/Users/icur/CursorProjects/FineTuneBase/data/alpaca_data_zh"
ds = Dataset.load_from_disk(dataset_path=dataset_path)
# print(ds)


# 分割数据集
ds = ds.train_test_split(100, seed=42)      # 100 代表从ds中分割出100个样本作为验证或测试数据
# print(ds)


# 数据预处理
model_dir="/Users/icur/CursorProjects/FineTuneBase/model_cache/glm"
tokenizer = AutoTokenizer.from_pretrained(model_dir)

def process_func(examples):
    contents = ["摘要生成: \n" + e for e in examples["contents"]]
    inputs = tokenizer(contents, max_length=384, truncation=True, padding="max_length", return_tensors="pt")      # 对contents字段内容进行词嵌入

    # 注意由于chatglm Transformers还没有收录, 对labels嵌入还需使用bulid_inputs_for_generation()来进行, 它是chatglm自带的方法
    # max_gen_length代表最大生成长度
    inputs = tokenizer.bulid_inputs_for_generation(inputs, target=examples["title"], padding=True, max_gen_length=64) # 对title字段内容进行词嵌入
    return inputs


# batch化
# 该模型输入为: input_ids, position_ids, attention_mask, labels
tokenized_ds = ds.map(process_func, batched=True, remove_columns=ds["train"].column_names)

# 查看它labels字段对应的tensor
# print(tokenized_ds["train"][0]["labels"])       # 最开始有很多-100, 就是代表prefix_embedding的部分

# 查看它的position
# print(tokenized_ds["train"][0]["position_ids"])   # 它有两个position embedding向量, 是glm特有的， partA和partB的





# 创建评估
# 由于chatglm是自定义的评估，先不使用transformers的评估实现，等到最后训练完之后在进行评估


# 配置训练参数
args = Seq2SeqTrainingArguments(
    output_dir="",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    logging_steps=8,
    num_train_epochs=1
)


# 创建训练器
trainer = Seq2SeqTrainer(
    args=args,
    model=model,
    train_dataset=tokenized_ds["train"],
    tokenizer=tokenizer    
)


# 开启训练
trainer.train()



# 模型推理测试
input_text = ds["test"][-1]["content"]
inputs = tokenizer("摘要生成: \n" + input_text + tokenizer.mask_token, return_tensors="pt")
inputs = tokenizer.bulid_inputs_for_generation(inputs, max_gen_length=64)
# 如果有gpu，则需移动到gpu上
# inputs = inputs.to("cuda")

# 获取模型生成的输出
output = model.genrate(**inputs, max_new_token=64, eos_token_id=tokenizer.eop_token_id, do_example=True)
tokenizer.decode(output[0].tolist())




# 使用测试数据，让model生成测试数据的摘要结果
model = model.eval()

def predict_test():
    predict = []
    with torch.inference_mode():
        for d in ds["test"]:
            inputs = tokenizer("摘要生成: \n" + d["content"] + tokenizer.mask_token, return_tensors="pt")
            inputs = tokenizer.bulid_inputs_for_generation(inputs, max_gen_length=64)
            # 如果有gpu，则需移动到gpu上
            # inputs = inputs.to("cuda")
            # 获取模型生成的输出
            output = model.genrate(**inputs, max_new_token=64, eos_token_id=tokenizer.eop_token_id, do_example=True)
            # 最后split()目的主要是为了获取开始特殊与解释特殊符直接的内容
            predict.append(tokenizer.decode(output[0].tolist()).split("<|startofpiece|>")[1].replace("<|endofpiece|>").strip())
            # 可考虑打印预测信息
            print("curID", len(predict))
    return predict

rest = predict_test() 

rouge = Rouge()
decode_preds = [" ".join(p) for p in rest]                      # 模型预测结果
decode_labels = [" ".join(l) for l in ds["test"]["title"]]      
scores = rouge.get_scores(decode_preds, decode_labels, avg=True)
scores_items = {
    "rouge-1": scores["rouge-1"]["f"],
    "rouge-2": scores["rouge-2"]["f"],
    "rouge-l": scores["rouge-l"]["f"],
}



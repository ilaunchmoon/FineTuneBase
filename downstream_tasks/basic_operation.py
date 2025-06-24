"""
    任何NLP的下游任务主要是如下步骤:

        1. 导入相关的包

        2. 加载数据集:  Dataset

        3. 数据集划分: Dataset 训练数据、验证数据、测试数据

        4. 数据预处理:  Dataset + Tokenizer, 转为能够输入LLM的格式前, 一般都要求做数据清洗工作

        5. 创建模型: Model

        6. 设置评估函数: Evaluate

        7. 配置训练参数: TrainingArgs

        8. 创建训练器: Trainer

        9. 开启训练: Trainer.train()

        10. 开启验证: Trainer.evaluate()

        11. 开启测试: Trainer.test()

    
        
    显存优化方法:

        以FP32数据类型为例

        模型权重: 4Bytes * 模型参数数量

        优化器状态: 8Bytes * 模型参数数量, 针对AdamW优化器, 一般是模型权重的 2倍

        梯度: 4Bytes * 模型参数数量

        前向激活值: 取决于序列长度、隐藏层维度、Batch大小等多个因素

        
    
    第一种: 不使用任何显存优化来进行训练

"""
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding


# 加载数据集, 当前只加载该数据集下的训练集: 仅仅设置为数据所在的根目录
dataset = load_dataset("/Users/icur/CursorProjects/FineTuneBase/data/Corpdb/", split="train")
dataset = dataset.filter(lambda x : x["review"] is not None)

# 划分数据集
datasets = dataset.train_test_split(test_size=0.1)      # 此时从上面dataset划分出训练和测试数据集


# 数据预处理: 主要是结合tokenizer转变为可以输入到LLM的数据格式
tokenizer = AutoTokenizer.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model/bert-base-chinese")
def process_data(example):
    tokenized_data = tokenizer(example["review"], max_length=128, padding="max_length", truncation=True)
    tokenized_data["labels"] = example["label"]
    return tokenized_data

tokenized_data = datasets.map(process_data, batched=True, remove_columns=datasets["train"].column_names)



# 创建模型
model = AutoModelForSequenceClassification.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model/bert-base-chinese")


# 创建评估函数
acc_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def eval_metric(eval_predict):
    predictions, labels = eval_predict
    predictions = predictions.argmax(axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    acc.update(f1)
    return acc 


# 配置训练参数
train_args = TrainingArguments(
    output_dir="/Users/icur/CursorProjects/FineTuneBase/outputs/bert_base_chinese_down_tasks",
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,
    logging_steps=500,
    num_train_epochs=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    metric_for_best_model="f1",
    load_best_model_at_end=True 
)


# 定义训练器
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=eval_metric
)


# 模型开启训练
trainer.train()


# 开启验证
trainer.eval()


# 开启测试
trainer.predict(tokenized_data["test"])


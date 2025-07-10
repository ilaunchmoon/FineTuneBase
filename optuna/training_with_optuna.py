"""
    使用optuna超参数调优的核心

        1. 创建模型需要使用函数的方式来进行创建, 并将创建好的模型以返回值的方式返回出来

            def model_init():
                model = AutoModel.from_pretrain(model_dir)
                return model

        2. 使用trainer.hyperparameter_search()方式来启动训练, 并进行超参数调优

                hyperparameter_search()的参数配置说明

                    hp_space: 超参数调优的空间, 比如设置学习率的调优范围[2e-5, 1e-4]
                    compute_objective: 评估的目标是什么, 比如文本分类的指标为F1分数
                    n_trials: 调优的次数
                    direction: 调优的方式, 是最小化还是最大化, 这个和你设定的compute_objective有关系, 如果设定compute_objective是F1分数, 那么direction就是最大化, 如果设置是loss, 那么direction就是最小化
                    backend: 设定到底使用那种自动调优工具, 比如这里是使用的optuna
                    hp_name: 用于设置每次调优信息的记录保存路径

                注意: 完成采用默认值一样可以, 但是最好是设置它们的值


                关于hp_space的设置, 需要使用函数的函数来进行配置, 以字典的形式来配置你所需要调整的超参数和对应的选择范围

                    # 配置需要进行超参数调优的参数和对应的范围
                    def default_hp_space_optuna(trial) -> dict[str, float]:
                        return {
                            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),  # 设置学习率为自动调优的超参数, 和它的优化空间范围
                            "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),              # 设置训练轮次为自动调优的超参数, 和它的优化空间范围
                            "seed": trial.suggest_int("seed", 1, 40),                                     # 设置随机种子值次为自动调优的超参数, 和它的优化空间范围
                            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32, 64]),  # 设置训练批次为自动调优的超参数, 和它的优化空间范围
                            "optim": trial.suggest_categorical("optim", ['sgd', 'adam', 'adamw'])       # 自定义设置一些可选择的优化器
                        }


                


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
# 由于使用了optuna, 则使用函数的方式创建模型并返回
def model_init():
    model = AutoModelForSequenceClassification.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model/bert-base-chinese")
    return model


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
    model_init=model_init,
    args=train_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=eval_metric
)



# 配置需要进行超参数调优的参数和对应的范围
def default_hp_space_optuna(trial) -> dict[str, float]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
        "seed": trial.suggest_int("seed", 1, 40),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16, 32, 64]),
        "optim": trial.suggest_categorical("optim", ['sgd', 'adam', 'adamw'])       # 自定义设置一些可选择的优化器
    }

# 模型开启训练
# 使用超参数搜索的方式进行训练
trainer.hyperparameter_search(
    hp_space=default_hp_space_optuna,
    compute_objective=lambda x: x["eval_f1"],      # 使用f1分数作为调优目标
    direction="maximize",                          # 由于使用f1作为调优目标, 所以调优的方向是最大化f1分数
    n_trials=10                                    # 最多尝试次数为10次
)


# 开启验证
trainer.eval()


# 开启测试
trainer.predict(tokenized_data["test"])
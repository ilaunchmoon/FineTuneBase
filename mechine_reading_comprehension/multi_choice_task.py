"""
多项选择任务

        给定一个或多个文档P, 以及一个问题Q和对应的多个答案候选(Choice/Candidates), 输出问题Q的答案A, A是答案候选中的某一项

        本质: 因从多个候选中选择正确的答案输出，如果候选的个数是固定的，那么本质就是一个多分类任务

    
    数据预处理的方法

        输入数据:

            context  question  candidates[是一个列表，具有多个候选]  answer

            
        组装数据(假设具有4个候选)

            context  question  condidate1    answer
            context  question  condidate2    answer
            context  question  condidate3    answer
            context  question  condidate4    answer

            假设不足4个的, 使用pad填充即可, 也就是说组装数据, 组装为最大的候选个数, 不足最大的获选个数直接PAD填充
            
        获取输出预测数据

            讲上面4个数据预测的4个结果logits, 最后使用线性层聚合到4个维度, 最后再做softmax, 最后使用交叉熵计算, 得到最后choice


    模型选择

        AutoModelForMultipeChoice 多选项选择任务 



""" 
import torch
import evaluate
import numpy as np
from datasets import DatasetDict
from transformers import AutoModelForMultipleChoice, AutoTokenizer, Trainer, TrainingArguments


# 加载数据集
# 因C3数据已经分好了train、validation、test的数集，则可以直接使用DatssetDict.load_from_disk()来加载本地数据
data_cache_dir = r"D:\VSCodeProject\Ding\Pipeline\data\c3"
c3 = DatasetDict.load_from_disk(data_cache_dir)
# print(c3) # 由3个Dataset构成DatesetDict, 三个Dataset分别为test、train、validation, 每个数据集中的每一条数据中有如下字段: id  context   question  choice  answer

# 由于c3还有test集, test集合中每条数据中answer字段都是空, 先将test数据集弹出
c3.pop("test")
# print(c3)
print(c3["train"][:3])




# 数据预处理
model_cache_dir = ""
tokenizer = AutoTokenizer.from_pretrained(model_cache_dir)

def process_data(examples):
    """
        examples: 是如下一条条的dict的数据
        Dataset
            {
                "context": ""
                "question": ""
                "choice": ""
                "answers": ""
            }
    """
    context = []
    question_choice = []
    labels = []


    # len(examples["context"])通过其中一个字段来获取整个数据条目的长度
    # 需要讲context中所有元素拼接, 因为context是一个list
    for idx in range(len(examples["context"])):
        cxt = "\n".join(examples["context"][idx])
        question = examples["question"][idx]
        choices = examples["choice"][idx]       # 具有多个回答候选
        for choice in choices:
            context.append(cxt)                 # 将拼接好cxt添加到context中
            question_choice.append(question + " " + choice)             # 将问题和候选回答添加到question_choice中去

        # 针对未满4个候选的choice使用padding添加到4个
        if len(choices) < 4:
            for _ in range(4 - len(choices)):
                context.append(cxt)
                question_choice.append(question + " " + "未知")  # 使用"未知"来填充
    
        # 添加labels
        labels.append(choices.index(examples["answer"][idx]))

    # 仅对only_first第一个字段context进行截断
    tokenized_examples = tokenizer(context, question_choice, truncation="only_first", max_length=256, padding="max_length")
    # 由于上面是4个context、4个question、4个question_choice、4个answer组装在一起的
    # 现在需要将组装起来context、question、question_choice、answer每个字段都分割成对应的4个list
    # 即
    """
        {
            context: [1, 2, 3, 4, 5, 9, 10, 11, 1, 29, 11, 23, 14, 12, 5, 6],
            question: [1, 2, 3, 4, 5, 9, 10, 11, 1, 29, 11, 23, 14, 12, 5, 6],
            question_choice: [1, 2, 3, 4, 5, 9, 10, 11, 1, 29, 11, 10, 11, 1, 29, 11],
            answer: [5, 9, 10, 4, 5, 9, 10, 11, 1, 29, 9, 10, 11, 1, 29, 6]
        }

        分割成:
        {
            context: [[1, 2, 3, 4], [5, 9, 10, 11], [1, 29, 11, 23], [14, 12, 5, 6]]
            其余字段类似
        }
    """
    tokenized_examples = {k: [ v[i:i+4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
    tokenized_examples["labels"] = labels
    return tokenized_examples


tokenized_c3 = c3.map(process_data, batched=True)



# 创建模型
model = AutoModelForMultipleChoice.from_pretrained("")



# 创建评估函数
accuracy = evaluate.load("accuracy")

def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)




# 配置训练参数
training_args = TrainingArguments(
    output_dir="/Users/icur/CursorProjects/FineTuneBase/outputs/multichoice",
    per_device_train_batch_size=10,
    per_device_eval_batch_size=20,
    num_train_epochs=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=500,
    load_best_model_at_end=True 
)


# 创建训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_c3["train"],
    eval_dataset=tokenized_c3["validation"],
    compute_metrics=compute_metrics
)


# 启动训练
trainer.train()


# 自定义测试模型的pipeline(因为transformers中对多项选择任务没有现成的pipeline
"""
    核心: 自定义一个模型预测方法
"""

class MultiChoicePipeline:
    def __init__(self, model, tokenizer)->None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device

    # 输入预处理, 使得能够满足llm的输入要求
    def preprocess(self, context, question, choices):
        cs, qcs = [], []
        for choice in choices:
            cs.append(choice)
            qcs.append(question + " " + choice)
        return self.tokenizer(cs, qcs, truncation="only_first", max_length=256, return_tensors="pt", padding="max_length")
    
    # 预测过程
    def predict(self, inputs):
        inputs = {k:v.unsqueeze(0).to(self.device) for k, v in inputs.items()}          # 先升维度, 再移动到相同的device上
        return self.model(**inputs).logits


    # 模型输出后处理
    def postprocess(self, logits, choices):
        prediction = torch.argmax(logits, dim=-1)
        return choices[prediction]

    def __call__(self, context, question, choices):              # 为了直接使用MultiChoicePipeline()即进行调用
        """
            先对输入进行预处理
            将预处理的结果输入给llm进行预测
            对预测结果进行后处理
            输出最后的结果
        """
        inputs = self.preprocess(context, question, choices)
        logits = self.predict(inputs)
        rest = self.postprocess(logits, choices)
        return rest



# 开启测试
pipe = MultiChoicePipeline(model, tokenizer)
response = pipe("小明在北京上班", "他在哪里上班", ["北京", "上海"])     # 本来choice应该是有4个, 但是测试的时候没有batch的概念，则可以大于或小于4个



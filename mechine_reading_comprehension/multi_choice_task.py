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
from datasets import DatasetDict
from transformers import AutoModelForMultipleChoice, AutoTokenizer, Trainer, TrainingArguments


# 加载数据集
# 因C3数据已经分好了train、validation、test的数集，则可以直接使用DatssetDict.load_from_disk()来加载本地数据
data_cache_dir = "/Users/icur/CursorProjects/FineTuneBase/data/c3"
c3 = DatasetDict.load_from_disk(data_cache_dir)
# print(c3) # 由3个Dataset构成DatesetDict, 三个Dataset分别为test、train、validation, 每个数据集中的每一条数据中有如下字段: id  context   question  choice  answer

# 由于c3还有test集, test集合中每条数据中answer字段都是空, 先将test数据集弹出
c3.pop("test")
# print(c3)
print(c3["train"][:3])




# 数据预处理
model_cache_dir = "/Users/icur/CursorProjects/FineTuneBase/model_cache/chinese-macbert-base"
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
    labels = []                     # labels对应的原始数据中的 answer字段


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
    tokenized_examples = {k: [v[i: i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
    
    # 对labels进行处理
    tokenized_examples["labels"] = labels

    return tokenized_examples



# 使用部分数据来测试process_data()方法的是否正确
test_examples = c3["train"].select(range(10)).map(process_data, batched=True)
print(test_examples)  # 输出: Dataset({features: ['id', 'context', 'question', 'choice', 'answer', 'input_ids', 'token_type_ids', 'attention_mask', 'labels'], num_rows: 10})












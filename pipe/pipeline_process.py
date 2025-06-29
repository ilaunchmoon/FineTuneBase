"""
    详细解读pipeline的工作流程

        1. 初始化tokenizer

                tokenizer = AutoTokenizer.from_pretrained(model)
        
        2. 初始化model

                model = AutoModel.from_pretrianed(model)
        
        3. 数据预处理(文本token化), 本质就是通过tokenizer将文本进行token, 转换为特定张量类型
           一般的模型tokenizer将文本会转为三个张量: input_ids、attention_mask、labels 
           Bert模型一般还会有任务类型特殊张量, token_type_ids, 因为Bert模型具有MLM任务和NSL任何, 而这些任务是通过特殊的token来实现的

                input_text = "这是一段测试文本"
                inputs = tokenizer(input_text, return_tensors="pt")    一般支持返回pytorch或tensorflow形式的张量, pt代表pytorch张量, tf代表tensorflow类型的张量


        4. 模型做预测推理, 模型输出每一个token的出现的概率, 所以这一步输出的是概率

                logits = model(**inputs).logits       ** 代表将inputs做解包操作, 因为经过tokenizer处理之后的文本一般都会转为如下三个张量: input_ids、attention_mask、labels 


        5. 对模型预测的结果做后处理, 由于模型输出的是每个token的概率, 所以需要结合温度系数、top_k、top_p将对应的概率转为token字符

                pred = torch.argmax(torch.softmax(logits, dim=-1)).item()       # 这里简单按照最大概率原则将模型预测出的最大的token, 其实这里得到的是最大概率所对应的token在词汇表中的位置
                results = model.config.id2label.get(pred)                       # 最后需要将每个对应最大概率token转换成token字符

"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 加载模型和分词器
# 注意不能使用AutoModel, 否则model(**inputs)的返回结果无法使用 .logits 这个属性
model = AutoModelForSequenceClassification.from_pretrained("D:\\VSCodeProject\\Ding\\Pipeline\\model\\bert-base-chinese")       
tokenizer = AutoTokenizer.from_pretrained("D:\\VSCodeProject\\Ding\\Pipeline\\model\\bert-base-chinese")
print(model.config)

input_text = "天气不错"
inputs = tokenizer(input_text, return_tensors="pt")
print(inputs)

# 直接获取logits（维度为 [batch_size, num_labels]）
outputs = model(**inputs)
logits = outputs.logits  # 直接访问logits属性
print(logits)

# 计算预测结果（注意维度）
pred = torch.argmax(logits, dim=-1).item()  # dim=-1 在类别维度求argmax
print(pred)

# 获取标签映射
rest = model.config.id2label.get(pred)     # id2label, 可以通过print(model)输出模型结果来查看到底有哪些属性
print(rest)


"""
    pipeline就是将上面所有步骤封装到一起, 所以pipeline的具体过程就是上面那几步

"""

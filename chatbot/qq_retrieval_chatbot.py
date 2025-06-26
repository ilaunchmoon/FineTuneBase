"""
    任务: 使用question - question的匹配模式

    数据: SophonPlus/ChineseNlpCorpus
          title reply 两个字段
          其中 title 作为question字段、reply作为answer字段

    模型: 使用自定义双塔模型


"""
import torch
import faiss
import pandas as pd
from dual_model import DualModels
from transformers import AutoTokenizer, BertForSequenceClassification


# 加载数据
data_dir = "/Users/icur/CursorProjects/FineTuneBase/data/train_pair_1w.json"
data = pd.read_csv(data_dir)


# 加载模型
# 注意这里使用检查点文件
model_dir = "/Users/icur/CursorProjects/FineTuneBase/outputs/text_similarity_dumodel/checkpoint-800"
model = DualModels.from_pretrained(model_dir)
model = model.eval()


# 加载tokenizer
# 注意这里使用检查点文件对应的基座模型即可
tokenizer = AutoTokenizer.from_pretrained(model_dir)


# 将用户的查询使用向量表示
questions = data["title"].to_list()      # 先获取title字段作为question，并转成list类型
vectors = []                            # 预先定义为vectors作为question的向量表示存放区
with torch.inference_mode():
    for i in range(0, len(questions), 32):  # 32作为batch_size, 可以自行调整，代表每32条数据进行一次向量嵌入
        batch_sens = questions[i:i+32]
        inputs = tokenizer(batch_sens, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        inputs = {k:v.to(model.device) for k, v in inputs.items()}          # 转成device上
        vector = model.bert(**inputs)[1]                                    # 使用model中的bert进行词嵌入操作
        vectors.append(vector)                                              # 添加到vector中

vectors = torch.concat(vectors, dim=0).cpu().numpy()                        # 在dim=0维度上拼接再转成cpu设备上，再转成numpy格式

# 对问题向量数据库创建索引
index = faiss.IndexFlatIP(768)                                              # 768是词嵌入的维度
faiss.normalize_L2(vectors)                                                 # 对嵌入进行归一化, 否则无法使用cos相似度计算来度量相似计算


# 对用户的问题进行词向量嵌入编码
question = "寻衅滋事"
with torch.inference_mode():
    inputs = tokenizer(question, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    inputs = {k:v.to(model.device) for k, v in inputs.items()}          # 转成device上
    q_vector = model.bert(**inputs)[1]                                  # 使用model中的bert进行词嵌入操作

# 对用户问题进行归一化
faiss.normalize_L2(q_vector)


# 向量匹配
scores, indexs = index.search(q_vector, 10)         # 使用用户的问题的嵌入向量到向量数据库中去检索，检索得到向量数据库中与用户查询最为相似的文本嵌入向量，获取相似度分数和索引，10代表最为相似的top-10个
top_k = questions.values[indexs[0].tolist()]        # 通过相似度最高的索引信息到向量数据库中去获取到用户问题最为相似的问题


# 召回
# 直接使用匹配结果，效果很可能不好
# 对相似度高的top-k个匹配结果，然后将用户的问题向量和top-k匹配结果再输入到LLM中计算相似度得分，最后输出得分最高的匹配结果对应的答案即可
# 先加载模型, 由于用户问题和top-k匹配结果再次进行相似度得分计算
cross_model = BertForSequenceClassification.from_pretrained(model_dir)
cross_model = cross_model.eval()

# 对top-k个结果和用户问题再次输入到cross_model中
candidates = top_k[:,0].to_list()
ques = [question] * len(candidates)
inputs = tokenizer(ques, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
inputs = {k:v.to(model.device) for k, v in inputs.items()}
with torch.inference_mode():
    logits = cross_model(**inputs).logits.squeeze()         # 分数
    results = torch.argmax(logits, dim=-1)


candidate_answer = top_k[:, 1].tolist()
match_question = candidate_answer[results.item()]   # 最后通过与用户最为相似的问题top-k一起输入llm，通过得分来获取该问题对应的答案
final_answer = data["reply"].to_list()[results.item()] 
print(final_answer)






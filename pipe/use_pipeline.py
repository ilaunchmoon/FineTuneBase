import torch
from transformers import pipeline
from transformers import AutoModel, AutoTokenizer

# 第一参数是任务类型, 第二参数加载的模型, 此处使用本地加载的方式
pipe = pipeline(
    "text-classification",
    model="D:\\VSCodeProject\\Ding\\Pipeline\\model\\bert-base-chinese"
)
print(pipe("天气很差"))


# 或者分开加载model和tokenizer
model = AutoModel("D:\\VSCodeProject\\Ding\\Pipeline\\model\\bert-base-chinese")
tokenizer = AutoTokenizer("D:\\VSCodeProject\\Ding\\Pipeline\\model\\bert-base-chinese")

pipe1 = pipeline(
    "text_classification",
    model=model,
    tokenizer=tokenizer
)

print(pipe1("心情不错"))


# 如果需要指定cuda的GPU: 注意gpu使用device=0来设置, 是int类数值, 不是字符
# pipe2 = pipeline(
#     "text_classification",
#     model=model,
#     tokenizer=tokenizer, 
#     device=0 
# )

# print(pipe2("心情不错"))


# 如果具有多张卡, 直接使用device_map来自动将模型分层存放在多个GPU上
pipe3 = pipeline(
    "text_classification",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.float16       # 一般多卡会配合半精度
)

print(pipe3("心情不错"))
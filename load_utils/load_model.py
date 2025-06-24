from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

snapshot_download(
    repo_id="bert-base-chinese",  # 模型ID
    local_dir="/Users/icur/CursorProjects/FineTuneBase/model_cache/bert-base-chinese"
)


# 模型本地加载，可以直接使用上面下载好的模型加载, 直接从上面保存的路径下加载即可
model = AutoModel.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model_cache/bert-base-chinese")
tokenizer = AutoTokenizer.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model_cache/bert-base-chinese")

# print(model)
# print(tokenizer)


# 模型保存到本地的方法: 必须将模型权重文件和tokenizer文件必须保存到同一个文件夹下
# 其实 snapshot_download 已经将整个模型都下载了，但它会保存很多其他杂项文件，为了只保存模型权重文件和tokenizer文件
# 可以使用如下来操作来保存, 
model.save_pretrained("/Users/icur/CursorProjects/FineTuneBase/model/bert-base-chinese")
tokenizer.save_pretrained("/Users/icur/CursorProjects/FineTuneBase/model/bert-base-chinese")


# 从本地加载模型
local_model = AutoModel.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model/bert-base-chinese")
local_tokenizer = AutoTokenizer.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model/bert-base-chinese")

print(local_model)
print(local_tokenizer)
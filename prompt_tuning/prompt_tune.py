from transformers import AutoModel, AutoTokenizer

# 加载模型和tokenizer
model = AutoModel.from_pretrained("Langboat/bloom-1b4-zh", cache_dir="/Users/icur/CursorProjects/FineTuneBase/model_cache")
tokenizer = AutoTokenizer.from_pretrained("Langboat/bloom-1b4-zh", cache_dir="/Users/icur/CursorProjects/FineTuneBase/model_cache")


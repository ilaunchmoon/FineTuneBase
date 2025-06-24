import os
import gradio as gr
from huggingface_hub import snapshot_download
from transformers import pipeline, AutoTokenizer, AutoModel



# 指定本地模型存储路径
model_path = "D:\\VSCodeProject\\Ding\\Pipeline\\model_cache\\uer\\roberta-base-chinese-extractive-qa"

snapshot_download(
    repo_id="roberta-base-chinese-extractive-qa",
    local_dir=model_path,
    local_dir_use_symlinks=False
)

# 从本地加载模型
local_pipeline = pipeline(
    "question-answering",
    model=model_path,
    tokenizer=model_path
)

# 启动服务
gr.Interface.from_pipeline(local_pipeline).launch()

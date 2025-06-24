import torch
from accelerate import Accelerator
from transformers import AutoModel, AutoTokenizer, pipeline 


# 单机多卡需要配合加速器

accelerator = Accelerator(
    mixed_precision="fp16",             # 自动处理多卡混合精度计算
    cpu= False if torch.cuda.is_available() else True
)
import torch
import copy
import torch.nn as nn 
from torch.nn import functional as F

# Lora的线性层 
class LoraLinear(nn.Module):
    def __init__(self, 
                 base_linear:nn.Linear,         # 进行被替换的基座模型的线性层
                 r:int=8,                       # 低秩矩阵的秩
                 lora_alpha:int=16,             # 低秩矩阵的缩放因子
                 dropout_rate:float=0.1,        # lora线性层设置dropout_rate
                 eval_mode:bool=False,          # 是否为推理模式                 
                 )->None:   
        super().__init__()
        self.base_layer = copy.deepcopy(base_linear)        # 将基座模型的参数深拷贝一次
        self.r = r 
        self.lora_alpha = lora_alpha 
        self.eval_mode = eval_mode 
        self.dropout = nn.Dropout(dropout_rate)
    
        self.lora_A = nn.Parameter(torch.empty(self.r, base_linear.in_features), dtype=base_linear.dtype) 
        self.lora_B = nn.Parameter(torch.empty(base_linear.out_features, self.r), dtype=base_linear.dtype)
        
        # 初始化
        # 对A进行初始化: 使用高斯初始化
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)

        # 对B进行初始化
        # 如果是训练模型, B矩阵要初始化为0矩阵
        # 如果是推理或测试模型, B矩阵使用高斯初始化
        if self.eval_mode:
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)
        else:
            nn.init.zeros_(self.lora_B)

        # 将基座模型的参数都冻结住
        for param in self.base_linear.parameters():
            param.requires_grad = False

    def forward(self, x:torch.Tensor)->torch.Tensor:
        # 计算缩放系数
        scaling = self.lora_alpha / self.r 

        # 先对x进行一次dropout
        # 再让它与A相乘
        adjust_linear = F.linear(self.dropout(x), self.lora_A)

        # 上一步结果与B相乘
        adjust_linear = F.linear(adjust_linear, self.lora_B)

        # 最后基座模型参数和 BA相融合
        return self.base_laye(x) + scaling * adjust_linear
    


def replace_lora_linear(modules:nn.Module,
                        r:int=8,
                        lora_alpha:int=16,
                        dropout_rate:float=0.1, 
                        eval_mode:bool=False,
                        require_embed:bool=False,       # 是否需要将嵌入层也进行训练
                        require_norm:bool=False,        # 是否需要将norm层进行训练
                        require_lm:bool=False           # 是否需要将mlp中线性层进行训练
                        )->None:
    
    # 遍历模型中的每个子模块, 将它的线性层进行替换为lora_linear
    for name, child in modules.children():

        # 如果任何子层中的特殊单元: 如嵌入、norm、lm输出头等需要进行微调训练, 则需要将它解冻
        if any(s in name for s in ["embed", "norm", "lm_head"]):
            if "embed" in name:
                require_grad = require_embed
            elif  "norm" in name:
                require_grad = require_norm
            else:
                require_grad = require_lm
            
            for param in child.parameters():
                param.require_grad = require_grad

        elif isinstance(child, nn.Linear):      # 如果是线性单元, 则将它们替换成lora_linear
            # 声明一个lora_linear层
            lora_linear = LoraLinear(child, r=r, lora_alpha=lora_alpha, dropout_rate=dropout_rate, eval_mode=eval_mode)
            # 然后进行替换
            setattr(modules, name, lora_linear)

        else:
            replace_lora_linear(modules=child, r=r, lora_alpha=lora_alpha,
                                dropout_rate=dropout_rate, eval_mode=eval_mode, 
                                require_embed=require_embed, require_norm=require_norm, 
                                require_lm=require_lm)

        

        
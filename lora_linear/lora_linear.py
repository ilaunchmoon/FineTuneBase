import copy
import torch
import torch.nn as nn 
from torch.nn import functional as F 


class LoraLinear(nn.Module):
    def __init__(self,
                 base_layer:nn.Linear,
                 r:int=8,
                 lora_alpha:int=16,
                 dropout_rate:float=0.1,
                 eval_model:bool=False
                 )->None:
        super().__init__()
        # 注意必须使用copy深拷贝，否则self.base_layer修改了什么, 传入base_layer也随之更改, 因为直接赋值的方式本质是原来的一个引用, 会共享同一份存储空间
        # 但是在lora替换目标线性层的过程中, 本质只能替换目标线性层, 其他地方不能更改
        self.base_layer = copy.deepcopy(base_layer)           
        self.r = r 
        self.lora_alpha = lora_alpha
        self.dropout = nn.Dropout(dropout_rate)
        self.eval_model = eval_model

        # ΔW = B * A     (output_feat, input_feat) = (output_feat, r) * (r, input_feat)
        # 注意基座模型参数 W (output_feat, input_feat) 因为pytorch中默认输入张量在前, 权重矩阵在后 y = x * W^T + b, 所以W(out_features, in_features)
        self.lora_a = nn.Parameter(torch.empty(r, base_layer.in_features), dtype=base_layer.weight.dtype)
        self.lora_b = nn.Parameter(torch.empty(base_layer.out_features, r), dtype=base_layer.weight.dtype)


        # 初始化B和A, 需要结合是否为训练模式来初始化
        # 假设以B来进行零初始化矩阵, A作为高斯初始化矩阵
        # 但是在推理阶段是B和A都需要进行高斯初始化, 测试阶段就仅将A进行高斯初始化
        nn.init.normal_(self.lora_a, mean=0.0, std=0.02)

        # 如果为测试阶段, 则B矩阵也进行高斯初始化
        # 如果为训练阶段, 则B矩阵进行零初始化
        if eval_model:
            nn.init.normal_(self.lora_b, mean=0.0, std=0.02)
        else:
            nn.init.zeros_(self.lora_b)

        # 将原始基座模型线性单元都冻结住, 假设self.base_layer不使用深拷贝一份base_layer
        # 那么基座模型冻结, 这里LoraLinear单元中的self.base_layer也会冻结, 那还怎么启动替换的作用
        # 注意这里不是去冻结基座模型中的base_layer, 而是冻结从基座模型深拷贝过来的self.base_layer
        # 最后
        for param in self.base_layer.parameters():
            param.requires_grad=False

    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        #  W' = W +  scaling * (B * A)  其中 scaling = alpha / r
        scaling = self.lora_alpha / self.r                          # 缩放因子
        lora_adjust = F.linear(self.dropout(x), self.lora_a)        # self.dropout(x) * self.lora_a.t()  --> x * a.t()    <===>   a * x
        lora_adjust = F.linear(lora_adjust, self.lora_b)            # lora_adjust * self.lora_b.t()      --> (x * a.t() * b.t()) <==>  b * (a * x)
        return self.base_layer(x) + scaling * lora_adjust           # self.base_layer + W +  scaling * (B * A)
    


# 将基座模型中指定linear进行替换为lora_linear
# 第一步: 找到指定的要替换的linear
# 第二步: 将基座模型linear替换层lora_linear
# 注意: 需要对embed、norm、lm_head层的只需开启梯度, 不能替换为lora_linear, 因为它们不是nn.Linear()分别是 nn.Embedding(), nn.layer_norm()
# 而lm_head虽然是nn.Linear()但是不能直接替换为lora_linear, 它最多只需参与训练, 因为它和下游任务有关系
def replace_lora_linear(
       moduls:nn.Module,
       r:int=8,
       lora_alpha:int=16,
       dropout_rate:float=0.1,
       eval_mode:bool=False,
       embed_requires_grad=False,
       layer_norm_requires_grad=False,
       lm_head_requires_grad=False
)->None:
    for name, child in moduls.named_children():
        if any(s in name for s in ["embed", "norm", "lm_head"]):
            if 'embed' in name:
                require_grad = embed_requires_grad 
            elif 'norm' in name:
                require_grad = layer_norm_requires_grad
            else:
                require_grad = lm_head_requires_grad
            for param in child.parameters():
                param.requires_grad = require_grad

        elif isinstance(child, nn.Linear):
            lora_linear = LoraLinear(base_layer=child, r=r, lora_alpha=lora_alpha, dropout_rate=dropout_rate, eval_model=eval_mode)
            setattr(moduls, name, lora_linear)

        else:
            replace_lora_linear(moduls=child, r=r, lora_alpha=lora_alpha, 
                                dropout_rate=dropout_rate, 
                                eval_mode=eval_mode, 
                                embed_requires_grad=embed_requires_grad, 
                                layer_norm_requires_grad=layer_norm_requires_grad,
                                lm_head_requires_grad=lm_head_requires_grad)
        
    
        



# 使用nn.Linear()版本LoraLinear
class LoraLiner(nn.Module):
    def __init__(self, 
                 base_layer:nn.Linear,
                 r:int=8,
                 lora_alpha:int=16,
                 dropout_rate:float=0.1,
                 eval_mode:bool=False)->None:
        super().__init__()
        self.base_layer = copy.deepcopy(base_layer)
        self.r = r 
        self.lora_alpha=lora_alpha
        self.dropout=nn.Dropout(dropout_rate)

        # W = (out_features, in_features) 
        # 特别注意nn.Linear(in_features, out_features)是指得将一个x张量, 从(~, in_features) 变成(~, out_features)
        # 所以这里是(in_features, r)
        self.lora_a = nn.Linear(base_layer.in_features, r, bias=False)
        self.lora_b = nn.Linear(r, base_layer.out_features, bias=False)
        
        nn.init.normal_(self.lora_a.weight, mean=0.0, std=0.02)
        if eval_mode:
            nn.init.normal_(self.lora_b.weight, mean=0.0, std=0.02)
        else:
            nn.init.zeros_(self.lora_b.weight)

        # 将基座模型中深拷贝过来的参数全冻结住
        for param in self.base_layer.parameters():
            param.requires_grad=False

    def forward(self, x:torch.Tensor)->torch.Tensor:
        scaling = self.lora_alpha / self.r 

        adjust = self.lora_a(self.dropout(x))
        adjust = self.lora_b(adjust)

        return self.base_layer(x) + scaling * adjust
    

        
        
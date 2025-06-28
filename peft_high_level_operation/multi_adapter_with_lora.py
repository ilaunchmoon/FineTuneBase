"""
    使用同一个自定义基座模型和多适配器配置与切换

    
    应用场景:

        假设现在有n个下游任务需要微调, 且使用同一个基座模型进行微调, 那么需要配置n个微调适配器进行微调训练
        一种办法是n个不同下游任务配置n个微调适配器参数进行同时训练, 然后n个下游任务微调训练好的权重参数保存为n份
        另一种办法是n个不同下游任务一个个进行微调训练, 然后一个个保存微调好的权重参数


        现在问题是: 
            
            模型推理阶段, 如果检查到不同下游任务, 如何快速加载基座模型和对应的下游任务训练好的参数权重
            
            一种办法是n个任务每次都加载基座模型和对应该下游任务的微调好的参数, 进行模型推理, 一般不会这么做, 因为同一个基座模型加载了n次, 很费显存和内存
            
            另一种办法是只加载一次基座模型, 不同下游任务分别共享同一个基座模型, 然后根据不同的任务加载对应不同的微调权重参数, 这才是正确的做法


    

    操作步骤:

        step1: 自定义一个基座模型(可以不是自定义的模型)

        step2: 配置多个下游任务所对应的微调配置信息, 以lora微调为例, 配置多个lora微调配置信息

        step3: 结合不同的下游任务lora配置和基座模型保存微调好的权重参数:

               使用如下进行保存不同下游任务训练好的权重:

                    model_a = get_peft_model(model, lora_a)
                    model_a.save_pretrained("./model_a")                # 注意它只会保存微调训练好的权重参数, 不会和基座模型合并之后再进行保存

                    model_b = get_peft_model(model, lora_b)
                    model_b.save_pretrained("./model_b")                 # 注意它只会保存微调训练好的权重参数, 不会和基座模型合并之后再进行保存

                    model_c = get_peft_model(model, lora_c)
                    model_c.save_pretrained("./model_c")


        step4: 加载不同任务对应的微调参数, 主要是通过参数adapter_name来实现加载不同的任务权重参数
               并使用load_apater()的方式在某些任务权重基础上, 再加载其他任务对应的权重参数, 从而做到多个任务的权重参数共享同一个基座模型

                load_model_a = PeftModel(model, mode_id="./model_a", adapter_name="model_a")        # 此时是加载a任务对应权重
                load_model_ab = load_model_a.load_adapter("./model_b", adapter_name="model_b")       # 然后再基于已经加载任务a的权重基座上, 再加载任务b的权重, 从而实现了任务a和任务b共享同一个基座模型
                load_model_abc = load_model_ab.load_adapter("./model_c". adapter_name="model_c")      # 在基于已经加载任务a和任务b的权重基座上, 再加载任务c的权重, 从而实现了任务a和任务b、任务c共享同一个基座模型
                其余类似操作


        
        step5: 按照不同任务切换激活不同任务对应的适配器权重参数, 使用set_adapter()来实现

                比如load_mode_ab模型是具有任务a和任务b的权重, 如果只需激活任务b, 则使用如下:

                    load_model_ab.set_adapter("model_b")


            


        其余和正常训练、评估、推理模型一样



"""
import torch
import torch.nn as nn 
from peft import LoraConfig, get_peft_model, PeftModel



# 自定义模型
class Classifier(nn.Module):
    def __init__(self, in_size, hidden_size, num_labels) -> None:
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        
        # 将层定义为模型的属性，而不是在forward中创建
        self.classifier = nn.Sequential(
            nn.Linear(self.in_size, self.hidden_size),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_labels)
        )
    
    def forward(self, x) -> torch.Tensor:
        return self.classifier(x)
    


# 定义一个自定义模型的实例
model = Classifier(10, 20, 2)



# 获取自定义模型中有哪些参数是可训练的
print("模型参数名称:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape}")

# 根据上面获取到的该模型所有可训练参数设定为你想要进行微调的参数, 即设定为target_modules中的值
# 设定lora配置
lora_config1 = LoraConfig(
    target_modules=["classifier.0", "classifier.3"],  # 对应 Sequential 中的第0和第3个模块
    r=16,  # LoRA 的秩
    lora_alpha=32,  # LoRA 的缩放参数
    lora_dropout=0.1,  # LoRA 的 dropout
)


lora_config2 = LoraConfig(
    target_modules=["classifier.0"],  # 对应 Sequential 中的第0个模块, 第二个lora配置仅仅针对第0个模块进行微调
    r=8,  # LoRA 的秩
    lora_alpha=16,  # LoRA 的缩放参数
    lora_dropout=0.1,  # LoRA 的 dropout
)

# 加载不同lora配置信息
# 基座模型model和lora_config1组合
model1 = get_peft_model(model=model, peft_config=lora_config1)









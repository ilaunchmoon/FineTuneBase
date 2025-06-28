"""
    使用自定义适配器结合lora来微调自定义的模型

    操作步骤:

        step1: 自定义模型

        step2: 自定义模型的实例

        step3: 获取自定义模型实例的所有可训练参数

        step4: 从step3中找到你想微调的模型参数的模块或参数名称, 通过这些名称可以在微调配置target_modules=[]中设定你想要进行微调的参数模块即可

        step5: 使用step4配置好的参数和自定义的模型实例创建一个peft_model进行微调训练


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
lora_config = LoraConfig(
    target_modules=["classifier.0", "classifier.3"],  # 对应 Sequential 中的第0和第3个模块
    r=16,  # LoRA 的秩
    lora_alpha=32,  # LoRA 的缩放参数
    lora_dropout=0.1,  # LoRA 的 dropout
)

# 根据适配器参数创建微调模型peft_model
peft_model = get_peft_model(model=model, peft_config=lora_config)

# 将lora权重保存好
peft_model.save_pretrained("./lora_model")





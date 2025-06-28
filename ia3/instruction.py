"""
    IA3思想    

        Infused Adapter by Inhibiting and Amplifying Inner Activations(IA3)

        抑制和放大内部激活, 通过可学习的向量对激活值进行抑制或放大, 一般对注意力层中K、V以及FFN模块三部分的激活值进行调整, 进而做到微调模型的效果
        训练过程中同样冻结原始模型的权重, 只更新可学习的部分向量
        训练完成后, 将微调训练好的部分参数权重与原始基座模型进行合并



    IA3做法



    
    





"""
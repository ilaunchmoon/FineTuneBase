"""
    关于peft库的一些常用高阶操作


    1- 自定义模型适配器

        应用场景: 如果微调的基座模型或自定义的模型没有在peft库中, 需要自定义实现微调的适配器, 来适配基座模型进行微调



    2- 多适配器加载与切换

        应用场景: 如果使用同一个基座模型, 但是需要测试多种微调参数或微调方法, 如何灵活切换微调适配器


    3- 禁用适配器

        应用场景: 如何获取基座模型的输入, 而不是使用微调参数之后的模型输出, 那么需要禁用适配器



    关于peft几个常用类和方法的使用说明

    
        peft_model = get_peft_model(model=model, peft_config=loar_config)  
        将基座模型转换为PEFT参数高效微调模型, model是基座模型, peft_config是高效参数微调的配置信息, 比如lora等, 只有通过这个方法get_peft_model获得的模型才可以进行微调

        
        PeftModel.from_pretrained(model=model, model_id=peft_model_checkpoint_dir)
        从本地加载已预训练好的适配器权重
        model: 已通过 get_peft_model 初始化的 PEFT 模型,  注意这个model必须是PEFT模型, 不能是基座模型, 即必须是get_peft_model()返回的模型
        model_id: 包含预训练适配器权重的目录路径(如训练后保存的 LoRA 权重), 也就是微调好的权重参数检查点文件
        将目录中的适配器权重加载到 PeftModel 的适配器模块中, 覆盖初始化时的随机权重
        返回一个完整可用的 PeftModel (基础模型冻结 + 适配器权重加载完毕)


        peft_model.merge_and_unload()

        将适配器权重与基础模型权重合并, 并移除 PEFT 框架结构
        将 LoRA 适配器的低秩增量权重合并到基础模型的对应模块中
        移除所有 PEFT 相关的附加层和结构，恢复为原始模型架构
        返回一个标准 PyTorch 模型, 即基座模型参数和微调参数权重合并在一起了, 此时它就是一个和预训练模型一个类型的模型，它与基模型不同的是, 它参数经过微调，也不再包含 PEFT 组件, 也不是PEFT模型


        merged_model.save_pretrained(merged_model_dir)
        就是将上一步合并之后返回的合并模型进行本地保存
        将合并后的完整模型保存到磁盘
        保存后的模型是独立完整的, 无需 PEFT 库即可加载, 可直接用于推理部署


        PeftModel.save_pretrained(adpater_dir)
        如果是使用PeftModel进行save_pretrained保存, 那么保存的文件仅仅包含微调训练好的权重文件, 不包含任何基座模型的文件
        那么加载时就有加载基座模型和对应微调adpater_dir文件
        等效PeftModel.save_adapter(save_directory, "default")
        

        PeftModel.save_adapter()
        只保存 PEFT 特有的适配器权重(如 LoRA 的低秩矩阵), 不包含基础模型的原始权重
        那么加载时就有加载基座模型和对应微调adpater_dir文件


        微调的一般操作步骤如下:

            定义微调的参数PeftConfig, 有LoraConfig、LoraQConfig、PromptTuningConfig、PromptEncoderConfig、PrefixTuningConfig、IA3Config、AdaLoraConfig

            定义Peft模型, 需要结合基座模型和上一步配置的微调参数, 使用get_peft_model(model=model, peft_config)来初始化可以用于微调的训练的Peft模型

            配置训练参数TrainingArguments

            配置训练器, 结合定义Peft模型和配置的训练参数

            定义评估函数和评估器

            开启训练

            加载微调适配器权重和基座模型来进行微调模型推理, 一般会设计使用 PeftModel.from_pretrained(model=model, model_id="上面保存的检查点文件地址")  其实model_id就是微调的权重文件

            推理判断微调的结果, 可以考虑将微调权重和基座模型进行合并, 并卸载适配器组件, 得到和基座模型类型一样的模型, 只不过当前合并之后的模型里面含有微调权重, 需要使用 merged_model = PeftModel.merge_and_unload(), 返回的是merge模型

            最后将合并好的模型保存在本地, 使用merged_model模型保存在本地, merged_model.save_pretrained(merged_model_dir)
            
            
"""

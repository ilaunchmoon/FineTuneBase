"""
    Deepspeed是微软开发的支持分布式训练的深度学习优化库
    使用前请安装 pip install deepspeed

    Deepspeed改进优化

      无论是单卡还是分布式训练，最终的目的就是将模型参数进行更新优化
      但是DDP数据并行分布式训练, 会将模型给每个进程中的GPU完整地将模型复制一份
      然后在各自进程中进行前向和反向传播, 优化模型参数
      如上有一个明显的冗余问题: 即每个GPU都存在一个模型副本、同一个每个副本模型进行前向/反向传播都具有激活值、梯度值、优化器状态的冗余副本

      针对DDP的冗余缺点, Deepspeed进行了无冗余改进


    Deepspeed做法

    
    Deepspeed的缺点

        Deepspeed采用的ZeRO-1/2相较于DDP, 通信成本没有增加, 但是ZeRO-3的通信成本大约是DDP的1.5倍
        因此如果模型参数量小于100B, 一般选择ZeRO-2, 如果模型参数没有超过100B, 选择ZeRO-3训练的时间会长多

    
    实操方法:

        1. 使用accelerate config来配置Deepspeed

            step1: 终端中执行 accelerate config

            step2: 选择使用Deepspeed即可, 其他配置视情况进行配置

            step3: 最后使用accelerate launch 当前脚本.py 进行训练即可

            注: 使用accelerate配deepspeed 不用改任务代码

        
        2. 使用accelerate来完整配置Deepspeed

            step1: 终端中执行 accelerate config

            step2: 配置完整的 Deepspeed config文件

                   新建deepspeed.config文件, 文件名可以换

                    可进入 https://hugging-face.cn/docs/accelerate/usage_guides/deepspeed#google_vignette 查询如何配置

            step3: 最后使用accelerate launch 当前脚本.py 进行训练即可

            

        注意:

            使用ZeRO-3的注意事项:

                zero3_init_flag 用于加载大模型

                zero3_save_16bit_model 用于保存模型

                如果使用config.json来配置deepspeed, 则需要使用stage3_gather_16bit_weights_on_model_save, 对应训练的代码文件中用于推理的部分, 必须使用with torch.no_grad()

            
            使用deepspeed时配置有关于deepspeed的参数, 必须要于训练脚本中的Trainer设置的参数一致
        

    

"""
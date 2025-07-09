"""
    accelerate分布式训练集成库

        accelerate 集成了多种分布式计算架构, 如: DDP、FSDP、Deepspeed等, 它本身不实现分布式训练框架
        只是集成和提高了多种分布式计算的架构的接口, 方便开发者调用

    核心点

        1. 创建实例

            accelerator = Accelerator()
        
        2. 使用accelerator来进行loss的反向传播

            accelerator.backward(loss) 

        3. 验证评估的时使用accelerator提供各进程间通信获取汇总预测结果的方法

            accelerator.gather_for_metrics(pred, batch["label"])

        4. 打印所有训练或验证信息都可以使用accelerator的print()方法来进行打印, 可防止每一个训练进程都打印一边

             accelerator.print(f"epoch: {epoch} / {epochs + 1}, step: {global_step}, loss: {output.loss.item()}")

        5. 真正在执行训练的主函数中使用accelerator来启动训练

             accelerator.prepare(model, optimer, trainloader, validloader)

    
        6. 启动训练方法

             accelerate launch 当前脚本名.py 来启动训练
        
             
        注意:

            启动训练前, 可以在终端执行命令  accelerate config 来配置训练参数

            如: 

                (1) 选择使用本机还是远程服务器, 一般选择本机

                    In which compute environment are you running?
                    Please select a choice using the arrow or number keys, and selecting with enter
                    ➔  This machine
                    AWS (Amazon SageMaker)

                (2) 选择多GPU

                    Which type of machine are you using?                                                                                                                                                                            
                    Please select a choice using the arrow or number keys, and selecting with enter
                        No distributed training                                                                                                                                                                                     
                        multi-CPU                                                                                                                                                                                                   
                        multi-XPU                                                                                                                                                                                                   
                        multi-HPU                                                                                                                                                                                                   
                     ➔  multi-GPU                                                                                                                                                                                                   
                        multi-NPU                                                                                                                                                                                                   
                        multi-MLU                                                                                                                                                                                                   
                        multi-SDAA                                                                                                                                                                                                  
                        multi-MUSA                                                                                                                                                                                                  
                        TPU

                (3) 输入使用机器数, 默认为[1], 即默认为1台, 可根据实际资源来输入即可, 一般单机多卡都是使用默认值1, 多机可以自行输入

                    How many different machines will you use (use more than 1 for multi-node training)? [1]: 

                
                (4) 是否应该在运行分布式操作时检查错误? 这样可以避免超时问题, 但速度会变慢, 按照默认值确定即可

                    Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]:      
    
                (5) 你想使用torch优化你的脚本与?(是/否):, 按照默认值确定即可

                    Do you wish to optimize your script with torch dynamo?[yes/NO]:  


                (6) 是否要使用FSDP, 按实际来选择

                    Do you want to use FullyShardedDataParallel? [yes/NO]:

                (7) 是否要使用Megatron-LM, 按照实际来选

                    Do you want to use Megatron-LM ? [yes/NO]:

                (8) 让你选择选择使用多少张卡训练


                (9) 使用哪几张GPU来训练, 默认为1, 按实际来调整, 如果有8张卡, 可以使用 1, 3, 4, 7 的方式来指定其中几张卡来进行训练

                    How many GPU(s) should be used for distributed training? [1]: 

                (10) 是否使用混合精度训练

            
                注意⚠️: 配置好了上面那些参数之后, 它会以yaml方式保存起来, 并且会输出它保存的路径
                       后续可以直接通过该路径找到文件, 按情况自行修改, 然后通过执行参数配置文件的方式来进行训练
        

                       可以使用accelerate launch --help 来查询有关命令的说明文档


"""
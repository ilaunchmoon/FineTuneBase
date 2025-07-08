"""
    为什么需要DDP(distributed data parallel)

        因为DataParallel()无法真正做到数据并行训练, 缺点见 data_parallel_instruction.py


    DDP的步骤

        Step1: 使用多个进程, 每个进程都加载模型和数据

        Step2: 各个经常同时进行前向传播

        Step3: 各进程分别计算loss后, 进行反向传播, 更新梯度

        Step4: 各进程进行通信, 将各个进程更新的梯度, 在各卡上进行同步

        Step5: 各个进程分别更新模型参数



    DDP与DP的区别

        1. 省去了中心ParameterSever, 不用每个卡计算更新为梯度, 中心ParameterSever就和对应的卡进行一次通信并更新ParameterSever上的参数

        2. 省去了从中心ParameterSever加载模型, 然后将模型分发给其他卡的过程, 直接该用了多进程并发加载模型和数据

        3. 省去了从各GPU上汇总到中心ParameterSever, 然后更新中心ParameterSever, 再讲更新好的模型参数再一次分发给各个GPU的过程, 这个过程进行了多次GPU通信, 极大受限于带宽

    
    
    多机分布式训练的基本概念

        group: 进程组, 一个分布式任务对应一个进程组, 一般就是所有卡都在一个组里

        world_size: 全局并行数, 一般情况下等于总的卡数

        node: 节点, 可以是一台机器, 或一个容器, 节点内包含多个GPU

        rank(global_rank): 整个分布式训练任务内的进程序号

        local_rank: 每个node内部的相对进程序号


        
    实操

        核心点

            1. 必须进行DDP训练环境配置

                import torch.distributed as dist  
                dist.init_process_group(backend="nccl")

            2. 数据集划分的时候必须指定随机种子, 以确保每一个进程中的随机划分的结果是一致的, 否则因随机划分的结果不一致, 导致某些进程可以看到验证集的信息


            3. 创建数据加载器DataLoader时, 必须使用DistributedSampler()来实现让不同进程加载数据, 以实现多进程并发加载数据

                如
                    train_loader = DataLoader(trianset, batch_size=10, shuffle=True, collate_fn=collate_func, sampler=DistributedSampler(trianset))
                    valid_loader = DataLoader(validset, batch_size=64, shuffle=False, collate_fn=collate_func, sampler=DistributedSampler(validset))

                    
            4. 必须通过os.environ[“LOCAL_RANK”]的方式来指定当前DDP中那些可用的GPU, 凡是涉及到模型或数据转移到设备上的代码都需要按照如下操作

                model.to(int(os.environ[“LOCAL_RANK”]))

                batch = {k:v..to(int(os.environ[“LOCAL_RANK”])) for k, v in batch.items()}

            
            5. 加载模型是必须使用DistributedDataParallel()修饰模型, 使得模型能够进行DDP训练

                from torch.nn.parallel import DistributedDataParallel as DDP


            6. 在计算loss时, 必须对所有训练进程中的loss取平均来作为模型训练的loss值

                一般是使用all_reduce()的集合间通信方式来获取各个训练进程间的loss, 最后求它们的均值

                import torch.distributed as dist  
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)     


            7. 在计算预测正确个数时, 必须对所有训练进程中的预测正确的个数汇总, 然后求出所有进程中预测正确的个数作为最终的预测正常的个数

                 一般是使用all_reduce()的集合间通信方式来获取各个训练进程间的正确个数, 最后求它们的总数

                 import torch.distributed as dist  
                 dist.all_reduce(corr_num, op=dist.ReduceOp.SUM)


            8. 运行脚本时, 必须指明一个进程数的参数 --nproc_per_node=进程数, 必须执行如下命令
               进程数根据你有多少张卡来决定, 有几张卡就设置为多少

                torchrun --nproc_per_node=进程数 当前脚本名.py



        
            
            





"""
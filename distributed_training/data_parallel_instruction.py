"""

    DP数据并行的原理

        即在每个GPU上复制一份完整的模型, 然后每个GPU上训练的数据不同, 所有GPU训练完后合并起来
        要求是每个GPU不仅能够存放完整的模型, 而且还能满足该模型执行完整的训练

    
    Pytorch的实现方法

        nn.DataParallel()实现数据并行方法

    
        
    DP训练的步骤

        Step1: 确定一个主GPU, 一般都是GPU0用于加载model和batch训练数据

        Step2: 将batch数据从GPU0均分到其他卡上

        Step3: 将model从GPU0复制到其他卡

        Step4: 除GPU0, 其他各卡进行前向传播

        Step5: GPU0收紧其他所有各卡上的输出, 并计算Loss

        Step6: 将Loss从GPU0分发到各卡, 然后各卡都进行反向传播, 并计算梯度

        Step7: GPU0收集各卡的梯度, 进行汇总

        Step8: GPU0更新模型的参数

    
        典型: ParameterServer-WorkNode架构, 即GPU0就是ParameterServer就是更新模型的中心, 其他各卡就是WorkNode, 真正在进行训练的工作节点


        
    DP实操代码

        见 data_parallel.py


        核心点:

            1. 创建的model, 需要使用 model = nn.DataParallel(model, device_ids=None)定义, 使其能够使用多卡进行训练

            2. 训练时的loss需要做标量处理

                模型进行反向传播时必须要求loss为标量, 否则无法进行反向传播计算

                由于PS-Work架构的多卡训练GPU0会将其他所有卡上计算的loss进行汇总, 此时返回给GPU0上的loss是一个向量
                分别代表除GPU0之外的所有卡上的loss值, 如 loss = torch.tensor([loss1, loss2, ..., lossn])
                为了让loss为一个标量, 只需将train()代码中所有使用到loss的地方都使用loss.mean()来替换即可

            
            注意: 以上是针对你自己实现train()等方法需要修改的地方, 如果你是transformers中自带的Trainer()结合TrainingArguments()训练器来训练
                 无需修改任何代码, 就可以实现数据并行训练, 它会自动识别当前环境下有多个GPU, 进而自动实现数据并行训练
    
                
    DP的问题:

        pytorch中nn.DataParallel()的实现数据并行的方式, 在数据批次较大的时候才能明显表现出训练优势, 并且 pytorch中nn.DataParallel() 速度提高并没有其他分布式加速包来得快, 后续会介绍

        问题1: 单线程、多线程, 由于GIL锁的问题, 不能充分发挥多卡的优势
        问题2: 由于nn.DataParallel()的训练策略问题, 会存在一个主节点(即ParameterSevers)占用比其他节点高很多的问题, 即负载不均衡
        问题3: 效率低, 每次训练开始都需要重新同步模型, 即从各个工作节点中进行通信拉回它们的训练结果, 则受限于通信带宽, 这也是PS-WordNode架构最大的缺点
        问题4: 只适合单机多卡进行数据并行训练, 不适合多机多卡的真正分布式集群训练

    

        


"""
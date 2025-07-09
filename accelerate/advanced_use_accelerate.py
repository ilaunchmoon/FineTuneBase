"""
    进阶使用 Accelerate, 主要涉及如下几个方面

        1. 混合精度训练

            混合精度训练指得: 模型使用FP32加载, 加载完毕后复制出一个模型, 复制出来的模型转为一个半精度(FP16)的模型
                           使用FP16的模型进行前向传播, 然后FP16的模型前向传播结果复制一份转成FP32
                           然后使用FP32的模型进行反向传播, 包括梯度、优化值等都是32FP, 反向传播结果需要更新模型参数, 此时又转会FP16, 然后再次进行前向传播

            混合精度训练特点:

                一定会加速模型的训练

                不一定会降低训练的所占显存, 当训练的数据批次很多时, 激活值会占据很多显存, 而只有模型激活值占用很多显存的时候, 会降低显存占用


            accelerate开启方式:

                (1) 方式一
                    mixed_precision="bf16"代表启用混合精度训练
                    accelerator = Accelerator(mixed_precision="bf16")
                
                    
                (2) 方式二

                    终端执行 accelerate config
                    执行到配置 mixed_precision 参数时, 选择 bf16 或 fp16 都是开启混合精度训练

                    然后执行 accelerate launch 当前脚本.py 即可

                
                (3) 方式三

                    直接在终端通过显示命令来指定

                    accelerate launch --mixed_precision bf16 当前脚本.py



        2. 梯度累积功能

            梯度累积指得: 在有限资源硬件条件下模拟更大批次的训练效果, 即通过将训练前设定的batch, 分割为更新mini-batch
                        每对一个mini-batch进行完前向和反向传播, 先不急着通过梯度进行模型参数更新, 而且每个mini-batch累积起来
                        等到达到预先设定的累积梯度步数后, 再根据累积的梯度来更新参数

            
             accelerate实现梯度累积方法:

                第一步: 设定梯度累积的步数   accelerator = Accelerator(gradient_accumulation_steps=xx)

                第二步: 训练中, 添加 accelerator.accumulatr(model)梯度累积上下文
                       with accelerator.accumulatr(model):

                第三步: 配合 if accelerator.sync_gradients: 来按照每完成一个梯度累积更新后输出训练信息, 而是再使用训练完多少步进行一次输出
                        if accelerator.sync_gradients:
                            accelerator.print(f"")





        3. 实验记录功能

            实验记录需要结合可视化记录工具: Tensorboard、WandB、Aim、CometML、Visdom等, 以Tensorboard为例

            第一步: 安装Tensorboard

            第二步: 创建Accelerator实例时, 指定project_dir

                    参数log_with代表使用什么可视化工具, project_dir可视化的结果存放路径
                    accelerator = Accelerator(log_with="tensorboard", project_dir="xxx")

            第三步: 开启tracker进行记录
                    
                    accelerator.init_trackers(project_name="xx")


            第四步: 在需要记录的信息的地方使用accelerator.log()进行记录, 记录的信息为可视化工具的数据原料
                    
                    accelerator.log(f"epoch: {epoch} / {epochs + 1}, step: {global_step}, loss: {output.loss.item()}")

            第五步: 训练结束后, 关闭tracker记录

                    accelerator.end_training()

                    
            第六步: 启动tensorboard对记录的信息进行可视化

                    通过端口转发的方式启动, 然后再选择上面记录的训练信息来进行可视化
                    

            


        4. 模型保存功能

            单机训练完毕后保存使用: model.save_pretrained(save_dir)

            分布式训练完毕后保存使用:

                (1) 保证模型所有文件, 推荐使用

                    accelerator.unwrap_model(model=model).save_pretrained(save_directory=save_dir,                          # 模型保存路径, save_dir为你要保存的路径
                                                                          is_main_process=accelerator.is_main_process,      # 代表只保存主进程上的模型, accelerator.is_main_process用于判断主进程
                                                                          state_dict=accelerator.get_state_dict(model)      # model为要保存的模型, 和最前面的是同一个参数, 该参数代表要保存基座模型的所有文件和peft的模型(如有的话)
                                                                          save_func=accelerator.save
                                                                          )         
            
                    
                (1) 仅保存模型权重模型, 一般不要使用

                    # 注意保存是.safetensor的形式, 只会保存.safetensor文件, 不会保存模型其他config、tokenizer文件
                    # 并且如果使用了peft微调, 使用该方式保存它, 它不会分出adapter_model和基座model来保存, 只会保存一个整体的模型权重
                    accelerator.save_model(model, save_dir)        

                    


        5. 断点续训功能

            即因某些不可控因素导致模型训练中断, 要从中断前最后一次保存的基础上接着进行训练的功能

            第一: 按照训练步或epoch保存检查点文件

            第二: 加载中断最后一次检查点文件

            第三: 跳过最后一次检查点文件中已经训练的数据, 继续进行训练



            做法:

                (1) 保存检查点

                    accelerator.save_state()

                
                (2) 加载中断前的最后一次检查点文件

                    accelerator.load_state()

                    
                (3) 计算出需要跳过的轮数和步数

                    resume_epoch, resume_step

                
                (4) 根据计算出要跳过的轮数和步数, 进行跳过

                    accelerator.skip_first_batchs(train_dataloader, resume_step)

                    
                
                




"""
import time 
import torch
import pandas as pd
from torch.optim import Adam
from accelerate import Accelerator                                  # 用于分布式计算
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification



# 创建Dataset
class TextDataset(Dataset):
    def __init__(self, data_dir)->None:
        super().__init__()

        self.data = pd.read_csv(data_dir)
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]
    
    def __len__(self):
        return len(self.data)
    


# 划分数据集
# 采用多进程训练数据的, 如果每个进程都进行一次划分, 那么势必会导致各训练进程间的训练数据和验证数据出现重叠的情况，会让模型过拟合
# 那么必须在多进程划分数据集中, 必须保证每一个进程间的数据划分是一致的, 而且还有保证训练和验证数据之间没有重叠的部分
# 使用generator参数来实现, 同一个随机种子来确保每次随机划分是一致的
# generator=torch.Generator().manual_seed(42)
def prepare_dataloader(data_dir, model_dir):
    dataset = TextDataset(data_dir)
    trianset, validset = random_split(dataset, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(42))
    # 创建DataLoader
    tokenizer = BertTokenizer.from_pretrained(model_dir)

    def collate_func(batch):
        texts, labels = [], []
        for item in batch:
            texts.append(item[0])
            labels.append[item[1]]

        inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        inputs["labels"] = torch.tensor(labels)
        return inputs
    
    # 创建trian集和valid集的DataLoader
    train_loader = DataLoader(trianset, batch_size=10, shuffle=True, collate_fn=collate_func)
    valid_loader = DataLoader(validset, batch_size=64, shuffle=False, collate_fn=collate_func)

    return train_loader, valid_loader


def prepare_model_optimer(model_dir):
    # 创建模型和优化器
    model = BertForSequenceClassification.from_pretrained(model_dir)
    optimer = Adam(model.parameters(), lr=2e-5)
    return model, optimer 





def evaluate(model, valid_loader, accelerator:Accelerator):
    corr_num = 0                                # 预测正确的个数
    model.eval()                                # 开启模型测试模式
    with torch.inference_mode():                # 推理模式
        for batch in valid_loader:
            output = model(**batch)             # 前向传播
            pred = torch.argmax(output.logits, dim=-1)      # 计算logits
            # 使用gather_for_metrics()获取所有进程中预测的个数汇总
            # 避免因为进程间批次不同, 而被自动padding数据, 导致acc计算错误
            pred, refs = accelerator.gather_for_metrics(pred, batch["label"])
            corr_num += (pred.long() == refs.long()).float().sum()       # 统计出预测正确的个数来计算正确率: 先将pred和batch['labels']都转成long类型, 再比较是否相等, 相等为True, 否则为False, 所以最后要将bool转成float类型(即0.0和1.0), 最后再将所有数相加就是1的个数值, 也就是预测正确的个数
    return corr_num / len(valid_loader.dataset)                                            # 预测正确的总个数 / 测试集的总个数

                

def train(model, train_loader, optimer, valid_loader, accelerator:Accelerator, epochs=1, log_step=100):              # log_step代表每100step输出一个训练信息
    global_step = 0                             # 记录总的训练步数
    start_time = time.time()
    for epoch in range(epochs):
        model.train()                           # 训练模式    
        for batch in train_loader:              # 在训练集中训练
            with accelerator.accumulate(model):     # 开启梯度累积
                optimer.zero_grad()                 # 梯度清零
                output = model(**batch)             # 将数据输入到llm中, 进行前向传播
                loss = output.loss                  # 使用loss保存output.loss
                accelerator.backward(loss)          # 使用accelerator来反向传播
                optimer.step()                      # 反向传播后, 梯度更新

                if accelerator.sync_gradients:      # 如果每进行一个梯度累积完毕后更新, 再进行打印训练信息
                    # 总的step数加1
                    global_step += 1
                    if global_step % log_step == 0:     # 每运行100次打印一次信息
                        # 使用accelerator中all_reduce()来计算loss, 也能保证在各进程间进行取平均, "mean"代表所有进程中loss加和后取平均值
                        loss = accelerator.reduce(loss, "mean")
                        # 使用accelerator的print来进行打印, 防止每个进程中的训练信息都会打印
                        accelerator.print(f"epoch: {epoch} / {epochs + 1}, step: {global_step}, loss: {output.loss.item()}")    
                        
                        # 设置accelerator的tracker需要记录的信息, 即日志信息
                        accelerator.log(f"epoch: {epoch} / {epochs + 1}, step: {global_step}, loss: {output.loss.item()}")

                    # 保存模型
                    if global_step % 500 == 0:
                        accelerator.save_model(model=model, save_directory="/Users/icur/CursorProjects/FineTuneBase/outputs/accelerate_training")
                        accelerator.unwrap_model(model=model).save_pretrained(save_directory="/Users/icur/CursorProjects/FineTuneBase/outputs/accelerate_training",
                                                                              is_main_process=accelerator.is_main_process,
                                                                              tate_dict=accelerator.get_state_dict(model),
                                                                              save_func=accelerator.save
                                                                             )
        # 每训练完一次epoch进行验证一次
        acc = evaluate(model=model, valid_loader=valid_loader, accelerator=accelerator)
        # 使用accelerator的print来进行打印, 防止每个进程中的训练信息都会打印
        accelerator.print(f"epoch: {epoch} / {epochs + 1}, acc: {acc}, used time {time.time() - start_time}")    
        # 设置accelerator的tracker需要记录的信息, 即日志信息
        accelerator.log(f"epoch: {epoch} / {epochs + 1}, acc: {acc}, used time {time.time() - start_time}")

    # 训练完毕结束tracker
    accelerator.end_training()


def main():

    # 创建Accelerator()实例
    # mixed_precision="bf16"代表启用混合精度训练
    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=2, log_with="tensorboard", project_dir="/Users/icur/CursorProjects/FineTuneBase/accelerate/training_tracker")

    # 开启训练数据记录
    accelerator.init_trackers("advanced_use_accelerate_tracker_info")

    model_dir=""
    data_dir=""

    trainloader, validloader = prepare_dataloader(data_dir=data_dir, model_dir=model_dir)
    
    model, optimer = prepare_model_optimer(model_dir=model_dir)
    
    # 使用 accelerator 来用于分布式计算
    model, optimer, trainloader, validloader = accelerator.prepare(model, optimer, trainloader, validloader)

    # 将accelerator实例传入到train函数中, 便于accelerator对loss进行反向传播
    train(model=model, train_loader=trainloader, optimer=optimer, valid_loader=validloader, accelerator=accelerator)

    

if __name__ == "__main__":
    main()



"""
    Trainer是Transformers库中专门用于训练模型的组件, 集成了训练、评估、多卡训练等核心组件

    注意: 使用Trainer进行模型训练对模型的输入和输出是有限制的, 要求模型返回元祖类型或ModelOutput的子类
         如果输入中提供了labels, 模型也要求能够返回loss的结果, 如果模型返回是元组类型, 要求loss值为元组的一个值
    
         
    核心:

        通过TrainingArguments配置训练参数, 使用Trainer进行训练、评估、预测


    完成流程如下:

        1. 创建TrainingArguments

        2. 创建trainer

        3. 模型训练 trainer.train()

        4. 模型评估 trainer.evaluate()

        5. 模型预测 trainer.predict()


    
    详细解读TrainingArguments参数设置说明:

        output_dir=./                                   代表设置模型checkpoint保存地址

        num_train_epochs=3.0                            代表训练轮次

        per_device_train_batch_size=8                   代表每个设备训练时的批次
        per_device_eval_batch_size=8                    代表每个设备评估时的批次

        logging_steps=500                               代表每训练多少step输出一次训练信息
        eval_strategy=IntervalStrategy.NO               代表评估策略, 默认为None, 一般设置为epoch, 代表每训练一个epoch进行一次评估

        eval_strategy=step                              代表评估策略, 默认为None, 如果设置为step, 代表每训练多少step进行一次评估, 此时必须设置eval_steps的值, 否则报错
        eval_steps=400                                  代表每多少step评估一次, 默认为None, 这个必须结合eval_strategy=step来使用
        
        save_strategy=SaveStrategy.STEPS                代表采用什么策略来保存模型, 默认是按照每多少个step保存一次, 此时必须配合save_steps来使用, 若希望每一个epoch保存一次, 将它的值设为epoch
        save_steps=500                                  代表每多少步保存一次模型

        save_total_limit=None                           设置最多保存多个模型, 默认为None, 如果设置为3, 并且此时保存策略为epoch, 那么就只会保存最新的3个epoch模型(即模型训练完的最后3个epoch的模型)

        
        learning_rate=5e-05                             代表初始学习率的设置, 一般不需要你修改

        weight_decay=0.0                                代表通过惩罚模型权的大小​​ 来实现正则化, 将模型的权重也纳入到损失函数的优化过程中, 防止模型训练过拟合, 一般可以自定义设置为0.01
                                                        ​​常见设置范围：​​ 通常设置在 0.0 到 0.1 之间。更具体地说：
                                                        0.0: 完全关闭权重衰减(L2 正则化) 模型只优化原始任务损失
                                                        0.01 (1e-2): 一个非常常用的默认起始值或基准值
                                                        0.001 (1e-3): 另一个非常常见的值，尤其在较大的模型(如BERT, GPT上
                                                        0.0001 (1e-4):  较小的权重衰减，适用于非常大的模型或担心正则化过强时
                                                        0.1 (1e-1): 相对较强的权重衰减，通常用在较小的模型或者有明显过拟合迹象且数据量不大的情况。高于 0.1 的值相对少见

                                                        
        metric_for_best_model=None                      设置以什么标准来保存训练最好的模型, 可以设置为f1分数, 一般配合load_best_model_at_end=True使用, 即在最后训练完成后, 会保留f1分数最高的模型
        load_best_model_at_end=True                     代表最后保存最好评估指标分数最高的模型
        

        gradient_accumulation_steps                     代表每进行多少次batch训练之后, 再进行一次梯度更新, 即累加多少次batch训练之后的梯度, 再进行一次梯度更新, 用于节省训练显存
                                                        假设训练数据batch=1, gradient_accumulation_steps = 32, 即每训练32个batch之后, 即累加32次训练后, 再进行一次梯度更新
                                                        注意: 一般batch=1与验证批次都是设置为1, 配合 gradient_accumulation_steps > 1的数来设置
                                                             但是每次训练和验证一个批次, 那么会需要更长的训练时间

        gradient_checkpointing                          代表开启梯度检查点, 如果开启这个, 模型在加载完毕之后要执行model.enable_input_require_grads(), 否则会保存





    
    
"""
from transformers import TrainingArguments

training_args = TrainingArguments(output_dir="/Users/icur/CursorProjects/FineTuneBase/outputs/checkponits")
print(training_args)



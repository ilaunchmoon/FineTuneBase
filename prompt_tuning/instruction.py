"""

    prompt-tuning原理

        冻结住基座模型的所有参数, 在训练数据前增加一小段prompt, 仅仅是训练这一小段prompt的表示层(即prompt的embedding层), prompt存在两种形式: hard prompt 和 soft prompt

    
    hard prompt 

        即prompt的内容是人为设定的, 其内容是固定的


    soft prompt

        即prompt内容不是人为固定设置的, 模型需要自己去学习


    注意:

         hard prompt tuing 效果一般都比 soft prompt tuning更好, 因为 soft prompt tuning 是随机初始化一个 prompt, 它可能很难学习到与下游任务适配

         针对 soft prompt tuning 具有无法很好的适配下游任务的prompt
         p-tuning的解决方案: 将soft-prompt的具体内容使用一个MLP或LSTM作为一个小型编码器对它进行编码, 使得soft-prompt的具体内容能够利用该小型编码器学习到更加适配下游任务的表征

"""
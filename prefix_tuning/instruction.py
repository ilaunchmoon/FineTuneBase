"""
    Prefix-Tuning

        Prefix-Tuning相比P-tuing而言, P-tuning是将prompt添加在输入的embeding层, 然后将prompt的embedding层的输出结果, 再次输入到一个MLP或LSTM的编码层上
        经过编码层的编码和学习, 使得prompt能够得语义丰富和适配下游任务表征, 从而让它能够微调适配好下游任务, 即P-tuning不会将编码结果拼接的LLM中注意力机制模块中

        Prefix-Tuning的思想是使用额外前缀向量拼接到LLM中的注意力机制模块中每一层的K和V中的Post——key-values中
        作为​​注意力机制内部的上下文调控器​​, 前缀在每个层的注意力模块前注入信息，确保了引导信号​​贯穿整个模型的深度​​，避免了在浅层注入信号导致的稀释问题
        进而让模型通过标注数据训练学习到更加适配下游任务的能力




    


"""
"""
    Prefix-Tuning

        Prefix-Tuning相比P-tuing而言, P-tuning是将prompt添加在输入的embeding层, 然后将prompt的embedding层的输出结果, 再次输入到一个MLP或LSTM的编码层上
        经过编码层的编码和学习, 使得prompt能够得语义丰富和适配下游任务表征, 从而让它能够微调适配好下游任务, 即P-tuning不会将编码结果拼接的LLM中注意力机制模块中

        Prefix-Tuning的思想是将prompt添加在输入的embeding层, 并且将该编码层的结果作为前缀拼接到LLM中的注意力机制模块中每一层的K和V中的Post——key-values中
        即prompt的embeding层的编码结构做完kv-cache的一部分, 从而让每次注意力计算都能将prompt的编码结构作为生成下一个token的上文, 进而让prefix那部分prompt能够更好的适配下游任务



    


"""
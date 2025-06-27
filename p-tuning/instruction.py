"""
    P-tuning思想

        prompt-tuning中soft-prompt类型, 由于它的做法是随机赋值一段prompt内容, 那么该段随机prompt很有可能无法很好适配下游任务, 从而导致微调训练效果不好
        
        因此, P-tuning的做法是将该soft-prompt具体内容在进行词嵌入后, 将它的词嵌入向量输入到一个LSTM或MLP层中进行编码, 使得它能够学习到更加适配下游任务的prompt的内容
        而不再是直接使用一个随机soft-prompt内容经过词嵌入后就输入LLM去微调训练

    
    

"""
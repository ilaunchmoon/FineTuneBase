"""
    transformers中根据下游任务的不同, 会在预训练或基座模型后面添加一个适配下游任务的输出层, 该输出层称之为Model Head
    
    该框架支持如下任务的Model Head

        ForCausalLM:  如自回归生成任务, 即Decoder-Only结构

        ForMaskedLM:  如Bert预测随机Masked token的任务, 即Encoder-Only结构

        ForSeq2SeqLM: 如T5从文本到文本的任务, Encoder-Decoder结构

        ForMultipeChoice: 多项选择的任务

        ForQuestionAnswering: 问答式任务

        ForSequenceClassification: 文本分类任务

        ForTokenClassification: token级别的分类任务, 如NER命名实体识别的任务
    
"""

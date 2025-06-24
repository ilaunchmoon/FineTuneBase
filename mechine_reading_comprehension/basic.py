"""
    阅读理解的任务(Machine Reading Comprehension, MRC)

        让模型基于给定上下文和问题, 让模型结合上下文来回答问题, 分为完形填空、答案选择(类似英语阅读理解)、最后一种是片段抽取式: 问题(Q)的答案(A)在文档(P)中, A时P中的一个连续片段


     阅读理解的任务的评估指标

        精确匹配度(Exact Match, EM): 计算预测结果和标准答案是否完全匹配
        
        模糊匹配(F1): 计算预测结果和标准答案之间字符级别的匹配程度

                    F1 = (2 * Precision * Recall) / (Precision + Recall)

                    Precision: 看预测的结果中有多少个token在标准答案中

                    Recall: 看预测结果有多少token / 标准答案有多少个token

        注意:  一般都是使用模糊匹配的方法来度量


    
    数据预处理

        数据处理格式为如下:

                [CLS] Question [SEP] Context [SEP]

        如何确定答案的位置

                使用 start_pos / end_pos (token的起始和终止位置) 和 offset_mapping(偏移量) 来确定答案的位置


        如何解决Context(上下文)过长的问题

                直接截断, 虽然实现简单, 但是会影响模型训练的效果, 因为可能将正确答案的截断

                滑动窗口, 即每个窗口都会与前一个窗口有一部分重叠, 虽然也会丢失一部分上下文, 但是比直接截断好多了

    
                
    模型部分

        ModelForQuestionAnswering

        本质: 先使用Bert模型进行编码, 然后经过一个全连接分类层, 最后输出答案的位置信息strat_pos与end_pos的概率

"""



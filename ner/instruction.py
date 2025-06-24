"""
    解释NER任务

        目标: 使用llm识别输入中地名、机构名、人名、专有名词等

        任务要求: 
                
                要识别出属于哪一类实体, 即地名、机构名、人名、专有名词中的哪一类

                要识别出实体的起始位置, 即边界
        
        NER标注:

                即标柱输入序列中每个token到底属于哪一种类别的标准, 因为NER任务本质就是识别出每一个词或若干个词是否是一个实体(人名、地名、机构名、专有名词)
                但是有的分词会把一个分词分成多个子词, 所以还需要一个规则来标记某个分词是否为某个词的子词

                以IOBES标注为例

                    O-代表实体的外部(即非实体)     B-代表实体的开始   I-代表实体的内部(或 M-代表实体的内部)    E-代表实体的结束    S-代表一个分词是单独构成一个实体   以上就是标记分词是否为某个词的子词前缀标记

                    如:

                        O 表示该token为非实体

                        B-Person 代表该token为人名实体的起始token

                        I-Person 代表该token为人名实体的中间token

                        E-Person 代表该token为人名实体的结束token

                        
                        LeBron James goes to the Hospital for his leg

                        假设上面序列分词之后如下:

                        Le       Bron      James      go     es      to       the       Hos          potal        for         his         leg

                        那么可能的NER标注为:

                        Le       Bron      James      go     es      to       the       Hos          potal        for         his         leg

                        B-Person I-Person  E-Person   O      O       O         O        B-Location   E-Location     O          O           O

                        
                        即: 一个完整的命令实体, 一定是具有完整 B-I-E, 或者 B-E, 或 S, 才能是一个完整的实体

                    






        模型:

              必须使用ModelForTokenClassification的token级的分类模型, 因为NER任务本质是对token进行分类

        
        模型评估:

            token级的分类任务, 即NER任务需要使用额外的评测指标

            安装 pip install seqeval 来进行评估


"""

# 
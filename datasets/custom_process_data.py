"""
    复杂自定义数据加载和处理

        1. 要求继承datasets.GeneratorBasedBuilder类来重写如下几个重要方法

            def _info(self)->DatasetInfo:
                
                该方法要做到如下：

                    定义数据集的信息，数据集中所有字段都要进行定义, 即一条完整的数据条目有哪些字段
                    一般description:str="用于描述数据集的信息", 需要有但是不重要
                    
                    如下就是定义一个完整数据条目的所有信息和层级关系, 一般都是使用字典的形式来定义, 它也会被返回
                    return features=datasets.Features(
                        {
                            "id": datasets.Value("string"),         # 具有id字段, 该字段的值类型为string
                            "context": datasets.Value("string"),    # 具有context字段, 该字段的值类型为string
                            "question": datasets.Value("string"),   # 具有question字段, 该字段的值类型为string
                            "answers": datasets.features.Sequence(  # 具有answers字段, 这个字段又是一个字典类型, 所以必须使用 datasets.features.Sequence来定义, 因为字典内部可能多个键值对
                                {
                                    "text": datasets.Value("string"),              # 具有text字段, 该字段的值类型为string
                                    "answer_start": datasets.Value("int32")        # 具有answer_start字段, 该字段的值类型为int32
                                }
                            )
                        }
                    )
            

            def _split_generators(self, dl_manager:DownloadManager):
                
                该方法具有如下要求: 主要是用于datasets中split参数功能的实现，方便进行数据集分割为训练集或测试集或验证集
                                   即主要功能是为了实现如何进行分割数据集合
                
                dl_manager: 

                返回参数类型:

                    [datasets.SplitGenerator]

                # name=datasets.split.TRAIN代表划分为训练数据集
                # gen_kwagrs是配合下面_generate_example来实现分割功能
                return [datasets.SplitGenerator(name=datasets.split.TRAIN, gen_kwagrs={"filepath: "你自定义数据集的本地地址"})]

                

            def _generate_example(self, filepath):
                该函数时最重要的函数, 它定义读取自定义数据、加载自定义数据，按照上面定义好字段进行组合数据，本质就是解析原始数据, 这里解析json复杂数据为例说明
                一般都是结合yeid来实现惰性加载，否则数据集太大一次加载充爆内存

                with open("文件路径", mode="r", encoding="utf-8") as f:
                    data = json.load(f)  # json文件解析
                    for example in data["data"]:        # 由于json文件真正的数据内容在该文件的data字段下, 所以只需读取该字段的内容即可
                        for paragraph in example["paragraph"]:       # data字段下具有：多个paragraph字段，
                            context = paragraph["context"].strip()                                        # paragraph字段下具有如下字段: context, qas字段, 而qas字段下又具有多个字段
                            for qa in paragraph["qas"]:         # qas下又具有: question、id、answers、answer_start多个字段
                                question = qa["question"].strip()
                                id_ = qa["id"]
                                answer_start = [answer["answer_start"] for answer in qa["answers"]]     # 由于具有多个, 所以采用list存放
                                answers = [answer["text"] for answer in qa["answers"]]                  # 由于具有多个, 所以采用list存放

                                # 特别注意: 这里返回的字段和字段直接的结果类型，必须和_info方法中定义的一模一样，否则会报错
                                yield id_, {
                                    "context": context,
                                    "question": question,
                                    "id": id_,
                                    "answer":{
                                        "answer_start": answer_start,
                                        "answers": answers
                                    }
                                }


        使用：

            使用的时候, 直接将带有上面来的脚本传入到load_dataset()中
            
            from datasets import load_dataset

            datasets = load_dataset("./脚本.py" split="train")
            
             
"""
import json
import datasets
from datasets import DownloadManager, DatasetInfo, GeneratorBasedBuilder, Features, Value, Sequence

class CMRDataset(GeneratorBasedBuilder):
    def _info(self) -> DatasetInfo:
        return DatasetInfo(
            description="CMRC2018 trial",
            features=Features({
                "id": Value("string"),
                "context": Value("string"),
                "question": Value("string"),
                "answers": Sequence({
                    "text": Value("string"),
                    "answer_start": Value("int32")
                })
            })
        )

    def _split_generators(self, dl_manager: DownloadManager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": "/Users/icur/CursorProjects/FineTuneBase/data/cmrc2018_trial.json"}
            )
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for example in data["data"]:
                for paragraph in example["paragraphs"]:
                    context = paragraph["context"].strip()
                    for qa in paragraph["qas"]:
                        question = qa["question"].strip()
                        id_ = qa["id"]
                        
                        # 提取答案和起始位置
                        answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                        answers = [answer["text"].strip() for answer in qa["answers"]]
                        
                        # 确保答案和起始位置数量一致
                        if len(answer_starts) != len(answers):
                            continue
                            
                        # 构建符合features定义的结构
                        yield id_, {
                            "context": context,
                            "question": question,
                            "id": id_,
                            "answers": {
                                "text": answers,
                                "answer_start": answer_starts
                            }
                        }


"""
    上面只是提供了分割训练集合, 如果进行test和validation集分割应该如下定义:
    
        def _split_generators(self, dl_manager: DownloadManager):
            # 定义数据集路径（根据实际情况修改）
            data_files = {
                "train": "/Users/icur/CursorProjects/FineTuneBase/data/cmrc2018_train.json",
                "validation": "/Users/icur/CursorProjects/FineTuneBase/data/cmrc2018_dev.json",
                "test": "/Users/icur/CursorProjects/FineTuneBase/data/cmrc2018_test.json"
            }
            
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"filepath": data_files["train"]}
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={"filepath": data_files["validation"]}
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={"filepath": data_files["test"]}
                )
            ]
"""
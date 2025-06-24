"""
    datasets是Transformers的组件, 可从本地或HuggingFace Hub中加载开源数据集, 处理数据集
    文档地址: https://huggingface.co/docs/datasets/index

    datasets基本功能:

        load_dataset: 在线加载数据集

        load_dataset: 加载某一项数据集的某一任务

        load_dataset: 按照数据集划分来进行加载, 如训练、验证、测试等

        index and slice: 查看数据集

        train_test_split: 数据集划分、训练集、测试集

        select and filter: 数据选取与过滤

        save_to_disk: 保存数据集到本地

        load_from_disk: 从本地加载数据集

"""

from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

# 1. 在线加载数据集
# 注意: 使用 load_dataset返回的是一个DatasetDict类对象
# data = load_dataset("madao33/new-title-chinese")       # 仅需在HuggingFace官网上的数据名即可， 此时会从官网进行在线加载

# 1.1 在线加载数据集并指定保存路径
# data = load_dataset("madao33/new-title-chinese", cache_dir="/Users/icur/CursorProjects/FineTuneBase/data_cache")

# 1.2 某些NLP任务是由多个子任务组成，而子任务都会由对应的数据
# data = load_dataset("madao33/new-title-chinese", "子任务数据集名", cache_dir="/Users/icur/CursorProjects/FineTuneBase/data_cache")


# 1.3 如果只需要某个数据集的训练集、验证集、测试集中某一个
# train_data = load_dataset("数据集名", split="train")      # 返回类型为Dataset类型, 不再是DatasetDict类型

# 1.4 如果只需要某个数据集的训练集、验证集、测试集中某一部分
# 如下代表加载数据集中训练数据集的前50%， 测试数据集的后30%, 注意必须使用百分数, 不要使用小数
# train_data_part = load_dataset("数据集名", split=["train[:50%]", "test[30%:]"])


# 2. 查看数据集
# data = load_dataset("madao33/new-title-chinese", cache_dir="/Users/icur/CursorProjects/FineTuneBase/data_cache")
# first_elem = data["train"][0]       # 由于data是DatasetDict类型, 则需要通过键和索引的方式获取第0条数据
# part_elem = data["dev"][:10]       # 取前10条数据

# 2.1 可通过数据集中的某个字段来获取
# title_part_elem = data["train"]["title"][:3]       # 获取训练数据就content字段下的前3条数据

# 2.2 可使用.column_names查看数据集字段信息
# data_column_name = data["train"].column_names

# 2.3 可使用.features查看字段具体信息
# data_col_info = data["train"].features


# 3.数据集划分
# 3.1 将训练集划分出训练集和测试集, 这里仅加载训练集
data = load_dataset("madao33/new-title-chinese", cache_dir="/Users/icur/CursorProjects/FineTuneBase/data_cache", split="train")
train_data, test_data = data.train_test_split(test_size=0.1) # 划分出0.1比例作为测试集


# 3.2 指定字段进行划分, 假设为分类问题的数据集, 一般为了保证类别均衡, 可以按照类别进行划分
# data_class_set = data_train.train_test_split(test_size=0.1, stratify_by_column="label")     # 注意要求 训练集中具有 "label"字段




# 4. 数据选取和过滤
# 4.1 使用select()选取， 输出是一个Dataset对象, 包含行数和字段名
data_select = data["train"].select([0, 1])


# 4.2 过滤数据fiter()集合lambda表达式
fiter_data = data["trian"].filter(lambda eample: "中国" in eample["title"])    # 选出字段title中包含 "中国 " 的数据



# 5. 数据映射: 使用map()函数, 一般要自定义映射函数集合map来实现
# 比如给某个字段中的值都添加一个前缀
def add_prefix(eample):
    eample["title"] = 'Prefix: ' + eample["title"]      # 给title字段下的值都添加一个前缀 "Prefix: "
    return eample

prefix_dataset = data.map(add_prefix)
prefix_dataset["train"][:10]["title"]   # 获取训练集中后，字段title对应前10条


# 5.1 实际使用方法
tokenizer = AutoTokenizer.from_pretrained("模型路径")
def process_func(example, tokenizer=tokenizer):
    model_inputs = tokenizer(example["content"], max_length=512, truncation=True, return_tensors="pt")       # 将字段content对应的内容进行tokenize
    labels = tokenizer(example["title"], max_length=32, truncation=True, return_tensors="pt")                # 将字段title对应的内容进行tokenize
    # 将文本的labels编码为tensor类型
    model_inputs['label'] = labels["input_ids"]     # 直接给model_inputs添加一个label字段对应tensor, 由于labels经过了tokenizer, 所以它包含了input_ids这个张量
    return model_inputs

# 结果batch来映射: 注意这里不仅有 input_ids, attention_mask，还有原始字段 title、content等
process_dataset = data.map(process_func, batched=True)   # batched要求有Fast类型Tokenizer, 如果没有Fast Tokenizer， 加速还可以通过num_proc来设置多线程加载
process_dataset = data.map(process_func, num_proc=4)     # 设置多线程加载, 但是必须要求在process_func得参数将tokenizer传入进去, 否则多线程无法直接保证线程安全


# 5.2 去除原始字段, 仅保留input_ids, attention_mask等能够输入到模型中字段
process_dataset = data.map(process_func, batched=True, remove_columns=data["train"].column_names)   # 移除原始字段, 通过data["train"].column_names获取原始字段


# 6. 保存处理好的数据: save_disk()
process_dataset.save_to_disk("保存到本地的硬件")


# 7. 加载保存好的数据: load_from_disk()
process_dataset = load_from_disk("保存数据的路径")






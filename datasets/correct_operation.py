from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer


# 1. 本地加载数据: DatasetDict类型, 其余加载部分或分割的部分都是Dataset类型
data = load_dataset("madao33/new-title-chinese", cache_dir="/Users/icur/CursorProjects/FineTuneBase/data_cache")
# print(data)

# 2. 按照数据集划分加载: 仅加载训练集, Dataset类型
train_dataset = load_dataset("madao33/new-title-chinese", cache_dir="/Users/icur/CursorProjects/FineTuneBase/data_cache", split="train")
# print(train_dataset)


# 3. 按照比例加载, 比如按照train数据的前50%加载, 注意仅能写百分数不能写小数, 表示百分数的数据不能有任何空格, Dataset类型
part_train_dataset = load_dataset("madao33/new-title-chinese", cache_dir="/Users/icur/CursorProjects/FineTuneBase/data_cache", split="train[:50%]")
# print(part_train_dataset)


# 4. 按照训练集和验证集的加载部分: 训练集加载前30%, 验证集加载后36%, 注意仅能写百分数不能写小数, 表示百分数的数据不能有任何空格, Dataset类型
part_train_valid_dataset = load_dataset("madao33/new-title-chinese", 
                                        cache_dir="/Users/icur/CursorProjects/FineTuneBase/data_cache", 
                                        split=["train[:30%]", "validation[36%:]"])      
# print(part_train_valid_dataset)


# 注意以下操作必须DatasetDict类型操作, 不可以使用以及分裂之后的数据Dataset来操作
# 5. 查看数据: 使用Dataset({})中键和索引来取, 必须使用DatasetDict类型来进行这种方式获取, 如果使用Dataset类型会报错
data = load_dataset("madao33/new-title-chinese", cache_dir="/Users/icur/CursorProjects/FineTuneBase/data_cache")
# print(data["train"][:2])        # 获取前两条

# 6. 使用数据集内部字段来取
# print(data["train"]["title"][:2])   # 前提数据集中具有title字段

# 7. 查看数据集字段信息
# print(data["train"].column_names)


# 这个划分操作可使用 Dataset来操作 进行划分
# 8. 按照字段进行划分, 比如分类问题中, 希望类别平衡, 所以可以按照字段名进行划分
train_dataset = data["train"]       # 获取data的训练集, 使用它来进行拆分为验证和训练集, 注意这里是 Dataset类型来操作, 不能是DatasetDict类操作
train_dataset, valid_dataset = train_dataset.train_test_split(test_size=0.1)


# 9. 按照字段进行划分, 如分类任务中, 可使用类别字段进行划分, 注意也必须使用Dataset类型来操作, 比如这个数据集中具有 "title" 该字段, 按照这个来分割
# 注意: 使用类别来分割必须要求对应的类别字段名为 ClassLabel 类型, 不能是其他类型, 比如如下title为Value类型会报错的
# train_split_by_label = data["train"].train_test_split(test_size=0.1, stratify_by_column="title")    
# print(train_split_by_label)



# 10. 过滤与选取
# 注意数据选取和过滤也必须要求为Dataset类型, 不能是DatasetDict类型
print(data["train"].select([0, 1, 2]))    # 选择前3条, 这里会输出是Dataset类型, 

# 11. 使用lamda表达是来作为过滤条件
# 注意数据选取和过滤也必须要求为Dataset类型, 不能是DatasetDict类型
# 比如这里选择字段title所对应的值中具有中国的
# print(data["train"].filter(lambda example: "中国" in example["title"]))  #  这里会输出是Dataset类型
# 如果要输出具体数据内容
# print(data["train"].filter(lambda example: "中国" in example["title"])[:5])  


# 11. 数据映射map函数: 注意这里使用的是DatasetDict类型
# 比如需要给字段对应的值前面都假设前缀: Prefixed: 这个字符, 一般需要定义映射函数
def add_prefix(example):
    example["title"] = "Prefix: " + example["title"]
    return example

prefix_dataset = data.map(add_prefix)
# print(prefix_dataset["train"][:2]["title"])


# 12. 结合tokenizer来进行数据预处理
tokenizer = AutoTokenizer.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model/bert-base-chinese")
def process_data(example):
    model_inputs = tokenizer(example["content"], max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(example["title"], max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# 可以将上面函数进行映射, 注意上面那个映射不仅仅包含可以输入到模型中的: input_ids, attention_mask, token_type_ids, labels, 还包含原始数据title和content字段对应的tensor
# 实际在输入到模型中, 原始字段对应的tensor一般是不会输入给model的, 接下来会解释如何去除原始字段
process_dataset_tensor = data.map(process_data)      
# print(process_dataset_tensor)


# 13. 结合参数batched=True来进行批处理, 但是批处理必须要求FastTokenizer
process_dataset_tensor = data.map(process_data, batched=True)      
# print(process_dataset_tensor)


# 14. 假设模型只有slowTokenizer, 那么可以使用num_proc参数来设置多线程处理
# 但是此时必须保证, process_data映射函数中将toknizer作为参数传入
# 不能将tokenizer放在外部, 因为外部是主线程区域, 和映射函数process_data不是在同一个线程环境
def process_data(example, tokenizer=tokenizer):
    model_inputs = tokenizer(example["content"], max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(example["title"], max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

process_dataset_tensor = data.map(process_data, num_proc=4)         # 4个线程
# print(process_dataset_tensor)


# 15. 去除原始字段对应的tensor
# 使用 remove_colnums参数来移除原始字段对应的tensor, 可以使用DictDataset类型、也可以使用Dataset类型
process_dataset_tensor = data.map(process_data, batched=True, remove_columns=data['train'].column_names)    # 通过设置remove_columns要移除的参数来移除指点的原始字段
# print(process_dataset_tensor)


# 16. 将上面预处理完成的数据存放在本地磁盘
# process_dataset_tensor.save_to_disk("本地存放路径")


# 17. 加载本地预处理好的数据
# process_dataset_tensor = load_from_disk("预处理好的本地存放路径")



# 



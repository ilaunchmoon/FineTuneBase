from datasets import load_dataset, load_from_disk

# 下载完整数据集（所有可用子集）
dataset_dict = load_dataset(
    "imdb",  # 数据集ID
    cache_dir="/Users/icur/CursorProjects/FineTuneBase/data_cache/imdb"  # 指定缓存目录
)
# print(dataset_dict)
train_dataset = dataset_dict["train"]  # 训练集
# print(train_dataset)
test_dataset = dataset_dict["test"]    # 测试集
# print(test_dataset)
unsupervised_dataset = dataset_dict["unsupervised"]     # 无监督数据集
# print(unsupervised_dataset)


# 一次性从本地目录下加载某个数据集中所有数据, 不是分train、test、validation来分别加载
# D:\VSCodeProject\Ding\Pipeline\data_cache\imdb\plain_text\0.0.0\e6281661ce1c48d982bc483cf8a173c1bbeb5d31
# \\imdb\\plain_text\\0.0.0\\e6281661ce1c48d982bc483cf8a173c1bbeb5d31

# 必须先保存到本地, 再进行加载，因为load_from_disk()只能加载DatasetDict()形式的数据
# 而load_dataset如果使用save_to_disk()保存的话, 那么数据就是一个缓存形式，里面有很多其他杂项文件
dataset_dict.save_to_disk("/Users/icur/CursorProjects/FineTuneBase/data/imdb")
full_data = load_from_disk("/Users/icur/CursorProjects/FineTuneBase/data/imdb")
print(full_data)


"""
    加载和预处理自定义的数据集

        1. 较为规范的数据加载

        2. 极为复杂的数据加载, 一般需要自定义加载脚本来实现加载和处理, 方法见 custom_process_data.py


"""
from datasets import load_dataset, load_from_disk, Dataset


# 1. 加载自定义数据集, 要求较为规范的数据集, 使用load_dataset(), 指定加载文件类型和存放数据的路径, 这里会加载路径下所有文件
dataset = load_dataset("csv", data_dir="D:\VSCodeProject\Ding\Pipeline\data\imdb")

# 1.1 或使用如下方式加载csv文件
dataset = Dataset.from_csv("D:\VSCodeProject\Ding\Pipeline\data\imdb")

# 1.2 如果需要加载路径下某几个文件: 使用data_files参数, 还可以结合split来指定到底加载什么数据集: 训练集还是测试集
dataset = load_dataset("csv", data_files=["./file1", "./file2"], split='train')


# 2. 加载其他类型数据, 比如加载pandas格式的数据类型, json, xml等datasets中的Dataset类都支持加载
# 2.1 假设数据定义为list, 必须使用Dataset.from_list()加载
data_list = [{"text": "12"}, {"text": "14"}, {"text": "112"}]
datalist_set = Dataset.from_list(data_list)






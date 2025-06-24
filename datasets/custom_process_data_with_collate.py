"""
    使用datasets中提供DataCollator进行加载

"""
from datasets import load_dataset

# 最好设置trust_remote_code=True, 否则每次运行都会中断询问你: 是否继续
dataset = load_dataset("/Users/icur/CursorProjects/FineTuneBase/datasets/custom_process_data.py", split="train", trust_remote_code=True)       # 利用自定义的脚本来加载训练集

print(dataset[0])


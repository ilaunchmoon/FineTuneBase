"""
    在torch.util.data的DataLoader如果需要按批次加载数据到模型, 一般会使用DataLoader中的参数collate_fn来进行操作
    
    datasets一样提供了封装好的DataCollate以实现类似的功能
    
"""

# 1. 先解释一个padding相关的 
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, AutoTokenizer

# 加载训练集
train_dataset = load_dataset("/Users/icur/CursorProjects/FineTuneBase/data/Corpdb", split="train")


# 2. 对原始数据中None过滤
train_dataset = train_dataset.filter(lambda x: x["review"] is not None)


# 3. 定义用于map的预处理函数
tokenizer = AutoTokenizer.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model/bert-base-chinese")
def process_data(example, tokenizer=tokenizer):
    inputs = tokenizer(example["review"], max_length=128, truncation=True)
    inputs["labels"] = example["label"]
    return inputs


# 4. 使用批次加载, 并清除原始字段
# 这里input_ids、attention_mask等能够输入到LLM得数据类型都还是list类型, 还不是tensor类型
train_tokenized_data = train_dataset.map(process_data, batched=True, remove_columns=train_dataset.column_names)
# print(train_tokenized_data[:2]) # 打印两个看看, 里面元素都是list

# 5. 使用DataCollatorWithPadding
# 创建这个Collator, 创建它一定要使用tokenizer模型
collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 使用 collator 来进行自动转为tensor和 自动padding, 注意的padding是动态的, 它不会每次都将padding到长度为上面tokenizer中设定的128维度, 目的是比较省内存资源
train_dataloader = DataLoader(train_tokenized_data, batch_size=4, collate_fn=collator, shuffle=True)




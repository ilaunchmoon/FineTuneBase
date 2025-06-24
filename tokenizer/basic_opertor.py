"""
    tokenizer详细操作过程
        
        1. 分词
            
            使用分词器将文本进行分词, 一般按照子词进行分词
        
        2. 构建字典

            将分词器分出的词(token)构建一个词典(词汇表), 目的是构建出token和与它们自己在词汇表中id进行一一映射
            构建字典映射不是必须的，假如是通过预训练模型来将token转为词向量，则一般不会做

        3. 数据转换

            通过构建好的字典，将分词处理后的数据做映射，将token(文本级信息)转为数字序列信息
        
        4. 数据填充与截断

            输入LLM的数据都是具有维度限制的，而且一般都是以批次的方式输入, 所以必须将每个batch的数据进行维度统一处理
            如果维度过大(即序列过长)做截断操作，如果维度过小(即序列过短)做padding操作

    
    tokenizer的操作

        from transformers import AutoModel, AutoTokenizer

        1. 加载和报错tokenizer

            tokenizer = AutoTokenizer.from_pretrained(模型的名称)

            sava_tokenizer = AutoTokenizer.save_pretrained(保存路径)

        2. 分词操作

            tokenize
        
        3. 查看字典

            vocab

        4. 索引转换
            
            convert_token_to_ids

            convert_ids_to_token
        
        
        5. 填充或截断

            padding
            
            truncation

        6. 其他操作

            input_ids, attention_mask, token_type_ids
        
"""


from transformers import AutoTokenizer


# 1 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model/bert-base-chinese")
# print(tokenizer)


# 2.保存 tokenizer
# tokenizer.save_pretrained("D:\\VSCodeProject\\Ding\\Pipeline\\demo_cache\\tokenizer_model\\bert_base_chinese_tokenizer")


# 3. 对文本进行分词操作
input_text = "你好"
tokens = tokenizer.tokenize(input_text)
# print(tokens)

# 4. 查看它的字典
# print(tokenizer.vocab)
# print(tokenizer.vocab_size) # 查看词汇表大写

# 5. 将分词进行索引转换
ids = tokenizer.convert_tokens_to_ids(tokens)
# print(ids)

token = tokenizer.convert_ids_to_tokens(ids)
# print(token)

# 5.1 可以使用ids直接转为token的string
str_token = tokenizer.convert_ids_to_tokens(ids)
# print(str_token)



# 6. 可以使用encode()方法直接将原始文本转为ids
# 注意: 直接使用分词器编码方式, 默认会有开始/结束等特殊标记token
ids = tokenizer.encode(input_text)
# print(ids)

# 忽略特殊标记
ids = tokenizer.encode(input_text, add_special_tokens=False)
# print(ids)


# 7. 分词ids解码器
# 解码器默认也会添加特殊token标记
str_token = tokenizer.decode(ids)
# print(str_token)

# 忽略特殊标记
str_token = tokenizer.decode(ids, skip_special_tokens=True)
# print(str_token)



# 8. 结合填充与截断操作: 一般都是结合最大长度来进行填充与截断
ids = tokenizer.encode(input_text, padding="max_length", max_length=20)
# print(ids)

# 8.1 截断操作: 即将padding部分截断掉
ids = tokenizer.encode(input_text, max_length=20, truncation=True)
# print(ids)


# 9. 使用encode_plus()会实现从原始文本到能够输入到模型数据转换包括: input_ids, attention_mask、token_type_ids等全部填上
inputs = tokenizer.encode_plus(input_text, padding="max_length", max_length=20)
# print(inputs)       # 此时inputs包含了input_ids, attention_mask、token_type_ids三个张量


# 10. 批次处理数据
# 10.1 假设当前输入3个批次
input_batch = [
    "我有一个希望",
    "我有一个梦想",
    "我有一个目标"
]

rest = tokenizer(input_batch, padding="max_length", max_length=10)
# print(rest)



# 11. Fast/Slow Tokenizer
# 注意: 上面所有Tokenizer默认都是Fast Tokenizer
# print(tokenizer)            # 输出结果: BertTokenizerFast, 后面有一个Fast后缀

"""
    Fast Tokenizer 和 Slow Tokenizer 区别:

        Fast Tokenizer 是由Rust实现的, token化后自动包含 input_ids, attention_mask 等张量, 速度比较快 return_offsets_mapping=True参数只能使用在FastTokenizer上

        Slow Tokenizer 是由Python实现的, token化后不包含 input_ids, attention_mask 等张量, 速度比较慢, 如果你发现数据处理比较慢, 检查是否是使用的SlowTokenizer


        # 需要使用use_fast = False来声明为Slow Tokenizer
        slow_tokenizer = AutoTokenizer.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model/bert-base-chinese", use_fast=False)

"""

slow_tokenizer = AutoTokenizer.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model/bert-base-chinese", use_fast=False)
# print(slow_tokenizer)       # 输出 BertTokenizer 此时没有Fast后缀



# 12. 使用tokenizer中的return_offsets_mapping参数, 该参数必须使用在FastTokenizer上, 
# 用法: return_offsets_mapping = True 代表启用 offset_mapping
# 用法: return_offsets_mapping = False 代表关闭 offset_mapping
# 功能: 用于标记某些子分词在原始分词中的起始位置, 在NER任务中很有用

# 此时分词 Dreaming 会被分割为 Dream ing两个分词
# 这它对应offset_mapping 可能是 (7, 12), (12, 15), 由此来表明(7, 12), (12, 15)是来源同一个分词, 并且分成了两个部分为: (7, 12), (12, 15)两个部分
input_text = "我的梦想, 你的Dreaming"           
tokens = tokenizer(input_text, return_offsets_mapping=True)
print(tokens)
print(tokens.word_ids())   # 输出每一个分词的位置索引


# 13. 自定义的Tokenizer模型的加载
# 假如自定义开发的分词器模型, 还没有上传到HuggingFace官方仓库时, 需要使用Tokenizer加载时, 必须使用turst_remote_code=True这个参数来进行加载或保存

# custom_tokenizer = AutoTokenizer.from_pretrained(model="模型名称或模型存放路径", trust_remote_code=True)
# print(custom_tokenizer)
# 保存自定义模型
# custom_tokenizer.save_pretrained("保存路径", trust_remote_mode=True)






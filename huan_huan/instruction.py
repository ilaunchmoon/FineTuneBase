"""
    关于对话式模型(Chat/Instruct版本)的LLM, 需要对tokenizer地方添加一些提示词

    提示词的内容一般包括如下部分:

        system: 给LLM设定一个和下游任务有关的角色
                设置它的需要做的事和不能做的事(如仅仅讨论物理方面的内容, 其他领域的内容不能讨论)
                设定它知识边界(如你的知识仅在2025年3月前, 关于之后的信息仅需回答: 无法回答)
                设定它​​时间上下文​​(即当前时间)
                设定它的行为规范(如你应该使用比较官方的语气来回答user的问题)

        user: 微调数据的上下文信息, 一般对应训练数据的instruct
              微调数据的输入, 一般对应训练数据的input


        assistant: 微调数据的输出, 一般对应训练数据的ouput
                   即训练数据的真实标签, 即人类期望模型针对user的查询的输出
                   并且该部分必须在最后添加上当前llm的结束标识符, 否则模型无法学习何时停止生成(导致无限续写)

        
        特别注意: 以上三个角色只有assistant的部分能够用于计算loss, 其余两个角色的token都不能用于计算loss


    提示设置和tokenize有两种办法

        (1) 使用手动拼接特殊标记token的方法

            见如下 process_func1()方法


        (2) 使用apply_chat_template()来进行, 这种办法比较适合用在多伦对话上, 不过单论对话也可以使用

            见如下 process_func2()方法
            process_func2()是一个全部使用add_special_token=True来自动添加特殊标记的方法, 针对多轮对话最好使用这些办法


"""

from transformers import AutoTokenizer


model_dir= None
tokenizer = AutoTokenizer.from_pretrianed(model_dir)


def process_func1(example):
    MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token, 因此需要放开一些最大长度, 保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    
    # add_special_tokens 不在开头加 special_tokens
    # 因为<|begin_of_text|>这些就是特殊tokens
    # 这部分必须给system和user角色已经关于assistant特殊标记token进行tokenizer
    instruction = tokenizer(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n现在你要扮演皇帝身边的女人--甄嬛<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['instruction'] + example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)  

    # response是作为assistant角色的提示词, 也就是训练数据的真实标签, 并且必须给assistant添加结束标识符<|eot_id|>
    response = tokenizer(f"{example['output']}<|eot_id|>", add_special_tokens=False)

    # 注意其实tokenizer.pad_token_id起到了结束标识token作用, 所以最好使用[tokenizer.eos_token_id]来替换
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为<|eot_id|>咱们也是要关注的所以补充为1, 如果不关注结束标识符, 模型会无休止的续写

    # 关于标签仅能使用assistant的token来计算loss, 所以其他部分token都使用-100来遮掩
    # 注意其实tokenizer.pad_token_id起到了结束标识token作用, 所以最好使用[tokenizer.eos_token_id]来替换
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  

    # 做截断
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }




def process_func2(example):
    MAX_LENGTH = 384
    
    # 1. 手动tokenize系统提示部分（保持原样）
    system_prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "Cutting Knowledge Date: December 2023\n"
        "Today Date: 26 Jul 2024\n\n"
        "现在你要扮演皇帝身边的女人--甄嬛<|eot_id|>"
    )
    system_part = tokenizer(system_prompt, add_special_tokens=False)
    
    # 2. 使用apply_chat_template处理用户输入部分（更灵活）
    user_messages = [
        {"role": "user", "content": example['instruction'] + example['input']}
    ]
    user_part = tokenizer.apply_chat_template(
        user_messages,
        tokenize=True,
        add_special_tokens=True,
        padding=False,              # 此时处理的是​​单个样本​​，padding 必须在批处理阶段统一进行, 所以先不进行padding
        return_dict=True
    )
    
    # 3. 手动tokenize助手响应部分（保持结束标记）
    response_text = f"{example['output']}<|eot_id|>"
    response_part = tokenizer(response_text, add_special_tokens=False)
    
    # 4. 组合所有部分
    input_ids = (
        system_part["input_ids"] +
        user_part["input_ids"] +
        response_part["input_ids"] +
        [tokenizer.pad_token_id]
    )
    
    attention_mask = (
        system_part["attention_mask"] +
        user_part["attention_mask"] +
        response_part["attention_mask"] +
        [1]
    )
    
    # 5. 生成labels（保持原始逻辑）
    labels = (
        [-100] * len(system_part["input_ids"]) +  # 系统部分
        [-100] * len(user_part["input_ids"]) +    # 用户部分
        response_part["input_ids"] +              # 助手响应（计算损失）
        [tokenizer.pad_token_id]                  # 最后pad token（计算损失）
    )
    
    # 6. 截断处理
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }



def process_func3(example):
    MAX_LENGTH = 384  # 保持原长度限制
    
    # 构建消息结构 - 单轮对话
    messages = [
        {
            "role": "system", 
            "content": "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n现在你要扮演皇帝身边的女人--甄嬛"
        },
        {
            "role": "user",
            "content": example['instruction'] + example['input']
        },
        {
            "role": "assistant",
            "content": example['output']
        }
    ]
    
    # 使用apply_chat_template一次性处理整个对话
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,          # 直接返回tokenized结果
        add_special_tokens=True, # 包含必要的特殊token
        padding=False,          # 不填充
        truncation=False,       # 不自动截断（手动处理）
        return_tensors=None,    # 返回列表而不是张量
        return_dict=True        # 返回字典格式结果
    )
    
    # 获取完整的input_ids
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    
    # 智能定位assistant内容的位置
    # 找到assistant标记的开始位置
    system_end_index = len(tokenizer.apply_chat_template(
        messages[:1],  # 仅系统部分
        tokenize=True, 
        add_special_tokens=True
    )["input_ids"])
    
    user_end_index = len(tokenizer.apply_chat_template(
        messages[:2],  # 系统+用户部分
        tokenize=True, 
        add_special_tokens=True
    )["input_ids"])
    
    # 生成labels
    labels = [-100] * user_end_index  # 系统+用户部分设为-100
    
    # 添加assistant内容部分（计算损失）
    assistant_part = tokenizer.encode(
        f"{example['output']}<|eot_id|>", 
        add_special_tokens=False
    )
    labels.extend(assistant_part)
    
    # 确保长度一致
    if len(labels) < len(input_ids):
        labels.extend([-100] * (len(input_ids) - len(labels)))
    elif len(labels) > len(input_ids):
        labels = labels[:len(input_ids)]
    
    # 最后添加一个pad token（如原始版本）
    input_ids.append(tokenizer.pad_token_id)
    attention_mask.append(1)
    labels.append(tokenizer.pad_token_id)
    
    # 截断处理
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
"""

    prompt-tuning原理

        冻结住基座模型的所有参数, 在训练数据前增加一小段prompt, 仅仅是训练这一小段prompt的表示层(即prompt的embedding层), prompt存在两种形式: hard prompt 和 soft prompt

    
    hard prompt 

        即prompt的内容是人为设定的, 其内容是固定的


    soft prompt

        即prompt内容不是人为固定设置的, 模型需要自己去学习


    
"""
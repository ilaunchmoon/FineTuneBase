"""
    evaluate是transformers中一个机器学习评估指标的库, 里面有很多常用的评估指标, 基本功能和操作如下:

        1. list_evaluation_modules: 查看支持的所有评估函数

        2. load: 加载评估函数

        3. inputs_description: 查看评估函数说明

        4. compute: 评估指标计算
                全局计算用: compute
                
                迭代计算用: add / add_batch

        5. combine: 计算多个评估指标

        6. radar_plot: 评估结果可视化

    注意: 不同的下游任务, 对应着不同的评估指标, 在官网 https://huggingface.co/tasks 中详细解释什么样的下游任务对应着那一种或几种评估指标的介绍


"""

import evaluate
import matplotlib.pyplot as plt
from evaluate.visualization import radar_plot


# 1. 查看支持的所有评估函数
# print(evaluate.list_evaluation_modules())

# 1.1 如果仅查看官方实现的, 不查看社区实现的, 使用include_community=False来过滤社区提供的评估函数
# print(evaluate.list_evaluation_modules(include_community=False))

# 1.2 如果查看每个指标的具体细节, 使用with_details=True
#print(evaluate.list_evaluation_modules(with_details=True))



# 2. 加载评估函数
# 比如加载准确率评估指标
accuracy = evaluate.load("accuracy")
# print(accuracy.description)             # 查看帮助文档, 会告诉你如何使用
# print(accuracy.inputs_description)      # 查看输入描述信息, 会告诉你如何输入
# print(accuracy)                         # 查看它的最为详细的文档说明

# 3. 全局计算
global_test_rest = accuracy.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0], normalize=False)  # 此时会输出预测对的个数
global_test_rest1 = accuracy.compute(references=[0, 1, 2, 0, 1, 2], predictions=[0, 1, 1, 2, 1, 0], normalize=True)  # 此时会输出预测对的正确率

# 4. 迭代计算, 即在训练时按照[真实标签, 预测标签]一对一对的添加进入再计算
for ref, pred in zip([0, 1, 0, 1], [1, 0, 0, 1]):
    accuracy.add(references=ref, predictions=pred)

# 4.1 迭代计算, 即在训练时按照[真实标签, 预测标签]一个批次一个批次的添加进入再计算
for ref, pred in zip([[0, 1], [0, 1]], [[1, 0], [0, 1]]):
    accuracy.add_batch(references=ref, predictions=pred)


# 5. 多个评估指标计算
binary_class_evaluate = evaluate.combine(["accuracy", "f1", "recall", "precision"])
combine_rest = binary_class_evaluate.compute(references=[0, 1, 0, 0, 1, 1], predictions=[0, 1, 1, 0, 1, 0])
print(combine_rest)


# 5. 评估结果对比和可视化
data = [
    
        {'accuracy': 0.616666666666666, 'f1': 0.1066666666666666, 'recall': 0.2666666666, 'precision': 0.166},
        {'accuracy': 0.716666666666666, 'f1': 0.2066666666666666, 'recall': 0.3666666666, 'precision': 0.966},
        {'accuracy': 0.816666666666666, 'f1': 0.3066666666666666, 'recall': 0.4666666666, 'precision': 0.366},
        {'accuracy': 0.916666666666666, 'f1': 0.4066666666666666, 'recall': 0.2666666666, 'precision': 0.566},
        {'accuracy': 0.4516666666666666, 'f1': 0.5066666666666666, 'recall': 0.1666666666, 'precision': 0.766}
    
]

model_names = ["qwen-3-1.8B", "llama-3-10B", "qwen-2.5-1.8B", "llama-2-10B","gpt-4-1.8B"]

plot = radar_plot(data=data, model_names=model_names)
plt.show()

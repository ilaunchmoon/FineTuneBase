from transformers.pipelines import SUPPORTED_TASKS


# 使用 SUPPORTED_TASKS获取transformers中支持的所有任务类型
for k, v in SUPPORTED_TASKS.items():
    print(k, v["type"])

# 使用 SUPPORTED_TASKS获取transformers中支持的所有任务类型和详细支持信息
for k, v in SUPPORTED_TASKS.items():
    print(k, v)


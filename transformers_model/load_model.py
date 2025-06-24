"""
    手动下载方式


        进入官网, 搜索到对应的模型

        下载对应pytorch.bin、token相关的所有文件, vocab.txt、special_tokens_map.json、config.json等文件, 不要下载.msgpack和.h5等文件(非pytorhc平台的权重文件)
        然后将这些文件放在一个文件夹下

        注意:

            早期Huggingface的模型都会将tensorflow、pytorch、msgpack等文件都放在一个仓库下, 导致你使用在线加载或下载都会将这些文件一起下载下来
            但是现在HuggingFace仓库一般只有.safetensors文件以及模型config的文件, 连vocab.txt文件都没有了, 所以做法是针对于早期的模型文件而言的手动下载方式
             

"""
from transformers import AutoModel, AutoConfig, AutoTokenizer


# 下载一个rbt3模型

# 1. 在线下载
# model = AutoModel.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model_cache/rbt3")


# 2. 加载本地rbt3文件
model = AutoModel.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model/rbt3")
# print(model)


# 3. 使用 AutoConfig获取模型的具体可操作参数
model_config = AutoConfig.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model/rbt3")
# print(model_config)  # 输出如下
"""
    BertConfig {
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "directionality": "bidi",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 3,
  "output_past": true,
  "pad_token_id": 0,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "position_embedding_type": "absolute",
  "transformers_version": "4.52.4",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 21128
}

"""


# 4. 模型调用
# 不带Model Head任务头的调用
input_text = "今天天气不错"
tokizer = AutoTokenizer.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model/rbt3")
inputs = tokizer(input_text, return_tensors="pt")
output = model(**inputs)
# print(output)


# 4.1 设置模型输出attention
inputs = tokizer(input_text, return_tensors="pt")
output = model(**inputs, output_attentions=True)
# print(output)


# 4.1 获取Bert最后编码last_hidden_state即对输入编码之后的输出
# print(output.last_hidden_state)


# 5. 带有Model Head任务头的调用
# 注意: Transformers中针对支持的任务都具备一个AutoModelFor任务头的类
from transformers import AutoModelForSequenceClassification
class_model = AutoModelForSequenceClassification.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model/rbt3")
# print(class_model)      # 注意如果不指明分类类别个数, 默认就是2分类, 即模型最后的输出是一个维度为2

class_output = class_model(**inputs)
#print(class_output)     # 这里会输出 loss, logits, 其中logits为二维张量, 由于这里没有将label输入, 则loss=None


# 5.1 通过num_labels来设定分类数,  比如设置5分类任务
five_class_model = AutoModelForSequenceClassification.from_pretrained("/Users/icur/CursorProjects/FineTuneBase/model/rbt3", num_labels=5)
five_class_output = five_class_model(**inputs)
print(five_class_output)   # 这里会输出 loss, logits, 其中logits为五维张量, 由于这里没有将label输入, 则loss=None






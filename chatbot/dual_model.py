import torch
from typing import Optional
from datasets import load_dataset
from torch.nn import CosineSimilarity
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification, BertModel, PretrainedConfig, BertPreTrainedModel, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding, pipeline



class DualModels(BertPreTrainedModel):  # 继承BertPreTrainedModel主要时为了使用.from_pretrained()方法
    def __init__(self, config:PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = BertModel(config)
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1. 获取sentenceA和sentenceB的输入
        senA_input_ids, senB_input_ids = input_ids[:0], input_ids[:1]
        senA_attention_mask, senB_attention_mask = attention_mask[:0], attention_mask[:1]
        senA_token_type_ids, senB_token_type_ids = token_type_ids[:0], token_type_ids[:1]
        
        # 2.获取sentenceA和sentenceB的embedding表示向量, 向量表示就是bert模型的输出
        senA_outputs = self.bert(
            senA_input_ids,
            attention_mask=senA_attention_mask,
            token_type_ids=senA_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        senA_pooled_output = senA_outputs[1]               # 即为A的最终Bert编码表示[batch, dim]

        senB_outputs = self.bert(
            senB_input_ids,
            attention_mask=senB_attention_mask,
            token_type_ids=senB_token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        senB_pooled_output = senB_outputs[1]               # 即为B的最终Bert编码表示[batch, dim]


        # 3. 计算相似度作为评估方法
        # 原本的分类模型会在bert输出后计算logits
        cos = CosineSimilarity(senA_outputs, senB_outputs)  # [batch]   

        # 计算相似度
        loss = None
        if labels is not None:
            loss 

# coding=utf-8

import os
import torch
import torch.nn as nn
from transformers import BertModel


import ipdb

class BaseModel(nn.Module):
    def __init__(self,
                 bert_dir,
                 dropout_prob=0.1):
        super(BaseModel, self).__init__()
        config_path = os.path.join(bert_dir, 'config.json')

        assert os.path.exists(bert_dir) and os.path.exists(config_path), \
            'pretrained bert file does not exist'

        self.bert_module = BertModel.from_pretrained(bert_dir)

        self.bert_config = self.bert_module.config

        self.dropout_layer = nn.Dropout(dropout_prob)

    @staticmethod
    def _init_weights(blocks, **kwargs):
        """
        参数初始化，将 Linear / Embedding / LayerNorm 与 Bert 进行一样的初始化
        """
        for block in blocks:
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0, std=kwargs.pop('initializer_range', 0.02))
                elif isinstance(module, nn.LayerNorm):
                    nn.init.zeros_(module.bias)
                    nn.init.ones_(module.weight)

    @staticmethod
    def _batch_gather(data: torch.Tensor, index: torch.Tensor):
        """
        实现类似 tf.batch_gather 的效果
        :param data: (bs, max_seq_len, hidden)
        :param index: (bs, n)
        :return: a tensor which shape is (bs, n, hidden)
        """
        index = index.unsqueeze(-1).repeat_interleave(data.size()[-1], dim=-1)  # (bs, n, hidden)
        return torch.gather(data, 1, index)


class bertForSequenceClassification(BaseModel):
    def __init__(self,
                 bert_dir,
                 dropout_prob=0.1,
                 ):
        super(bertForSequenceClassification, self).__init__(bert_dir=bert_dir,
                                               dropout_prob=dropout_prob)


        out_dims = self.bert_config.hidden_size

        self.classifier_oce = nn.Linear(out_dims, 7)
        self.classifier_ocnli = nn.Linear(out_dims, 3)
        self.classifier_tnews = nn.Linear(out_dims, 15)
        self.dropout = nn.Dropout(0.1)


        self.loss_fct = nn.CrossEntropyLoss()

        init_blocks = [self.classifier_oce, self.classifier_ocnli, self.classifier_tnews]

        self._init_weights(init_blocks, initializer_range=self.bert_config.initializer_range)

    def forward(self,
                token_ids,
                attention_masks,
                token_type_ids,
                task=None,
                labels=None):
        tasks = []

        # ipdb.set_trace()
        bert_outputs = self.bert_module(
            input_ids=token_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )
        if task:
            if not isinstance(task,list):
                assert task in ['oce', 'tnews', 'ocnli']

                # seq_out = bert_outputs[0]
                pooled_out = bert_outputs[1]
                pooled_out = self.dropout(pooled_out)
                num_labels = 0
                logits = None
                if task == 'oce':
                    logits = self.classifier_oce(pooled_out)
                    num_labels = 7
                elif task == 'ocnli':
                    logits = self.classifier_ocnli(pooled_out)
                    num_labels = 3
                elif task == 'tnews':
                    logits = self.classifier_tnews(pooled_out)
                    num_labels = 15
                outputs = (logits,)
                if labels is not None:
                    loss = self.loss_fct(logits.view(-1,num_labels), labels.view(-1))
                    outputs = (loss,) + outputs

                return outputs


            else:
                tasks = task
                # seq_out = bert_outputs[0]
                pooled_out = bert_outputs[1]
                pooled_out = self.dropout(pooled_out)
                num_labels = []
                logits = []
                for idx,task in enumerate(tasks):
                    row = pooled_out[idx]
                    logit,num_label = None,None
                    if task == 'oce':
                        logit = self.classifier_oce(row)
                        num_label = 7
                    elif task == 'ocnli':
                        logit = self.classifier_ocnli(row)
                        num_label = 3
                    elif task == 'tnews':
                        logit = self.classifier_tnews(row)
                        num_label = 15
                    logits.append(logit)
                    num_labels.append(num_label)
                outputs = (logits,)

                if labels is not None:
                    loss_l = []
                    for idx,label in enumerate(labels):
                        loss_l.append(self.loss_fct(logits[idx].view(-1, num_labels[idx]), label.view(-1)))
                    loss = torch.mean(torch.stack(loss_l))
                    outputs = (loss,) + outputs

                return outputs
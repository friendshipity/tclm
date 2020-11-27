# coding=utf-8

import os
import torch
import torch.nn as nn
from transformers import BertModel
import numpy as np
from src_final.utils.loss_utils import FocalLoss

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
                 weighted=False,
                 focal=False,
                 ):
        super(bertForSequenceClassification, self).__init__(bert_dir=bert_dir,
                                                            dropout_prob=dropout_prob)

        out_dims = self.bert_config.hidden_size

        self.classifier_oce = nn.Linear(out_dims, 7)
        self.classifier_ocnli = nn.Linear(out_dims, 3)
        self.classifier_tnews = nn.Linear(out_dims, 15)
        self.dropout = nn.Dropout(0.1)

        if focal:
            self.loss_fct = FocalLoss(alpha=0.25, gamma=2)
        else:
            self.loss_fct = nn.CrossEntropyLoss()

        self.weighted = False
        if weighted:
            self.weighted = True
            class_nums = [525, 7975, 808, 11210, 3657, 3896, 3657]
            cn = [1 / e * min(class_nums) for e in class_nums]
            weight_CE = torch.from_numpy(np.array(cn))
            self.loss_oce(weighted=weight_CE)

            class_nums = [1111, 4081, 4976, 3991, 5200, 2107, 4118, 3437, 5955, 3632, 3368, 4851, 257, 2886, 3390]
            cn = [1 / e * min(class_nums) for e in class_nums]
            weight_CE = torch.from_numpy(np.array(cn))
            self.loss_tnews(weighted=weight_CE)

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
            if not isinstance(task, list):
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
                    if self.weighted:
                        if task == 'oce':
                            loss = self.loss_oce(logits.view(-1, num_labels), labels.view(-1))
                        elif task == 'tnews':
                            loss = self.loss_tnews(logits.view(-1, num_labels), labels.view(-1))
                        else:
                            loss = self.loss_fct(logits.view(-1, num_labels), labels.view(-1))

                    outputs = (loss,) + outputs

                return outputs


            else:
                tasks = task
                # seq_out = bert_outputs[0]
                pooled_out = bert_outputs[1]
                pooled_out = self.dropout(pooled_out)
                num_labels = []
                logits = []
                for idx, task in enumerate(tasks):
                    row = pooled_out[idx]
                    logit, num_label = None, None
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
                    for idx, label in enumerate(labels):
                        if self.weighted:
                            if tasks[idx] == 'oce':
                                loss_ = self.loss_oce(logits[idx].view(-1, num_labels[idx]), label.view(-1))
                            elif tasks[idx] == 'tnews':
                                loss_ = self.loss_tnews(logits[idx].view(-1, num_labels[idx]), label.view(-1))
                            else:
                                loss_ = self.loss_fct(logits[idx].view(-1, num_labels[idx]), label.view(-1))
                            loss_l.append(loss_)
                        else:
                            loss_l.append(self.loss_fct(logits[idx].view(-1, num_labels[idx]), label.view(-1)))

                    loss = torch.mean(torch.stack(loss_l))
                    outputs = (loss,) + outputs

                return outputs


class GateOut(bertForSequenceClassification):
    def __init__(self,
                 bert_dir,
                 dropout_prob=0.1,
                 ):
        super(GateOut, self).__init__(bert_dir=bert_dir,
                                      dropout_prob=dropout_prob)
        self.gate1 = nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)
        self.gate2 = nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)
        self.gate3 = nn.Linear(self.bert_config.hidden_size, self.bert_config.hidden_size)

        init_gates = [self.gate1, self.gate2, self.gate3]

        self._init_weights(init_gates, initializer_range=self.bert_config.initializer_range)

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
            if not isinstance(task, list):
                assert task in ['oce', 'tnews', 'ocnli']

                # seq_out = bert_outputs[0]
                pooled_out = bert_outputs[1]
                pooled_out = self.dropout(pooled_out)
                num_labels = 0
                logits = None
                if task == 'oce':
                    gate = torch.sigmoid(self.gate1(pooled_out))
                    pooled_out = pooled_out * gate
                    logits = self.classifier_oce(pooled_out)
                    num_labels = 7
                elif task == 'ocnli':
                    gate = torch.sigmoid(self.gate2(pooled_out))
                    pooled_out = pooled_out * gate
                    logits = self.classifier_ocnli(pooled_out)
                    num_labels = 3
                elif task == 'tnews':
                    gate = torch.sigmoid(self.gate3(pooled_out))
                    pooled_out = pooled_out * gate
                    logits = self.classifier_tnews(pooled_out)
                    num_labels = 15
                outputs = (logits,)
                if labels is not None:
                    loss = self.loss_fct(logits.view(-1, num_labels), labels.view(-1))
                    outputs = (loss,) + outputs

                return outputs


            else:
                tasks = task
                # seq_out = bert_outputs[0]
                pooled_out = bert_outputs[1]
                pooled_out = self.dropout(pooled_out)
                num_labels = []
                logits = []
                for idx, task in enumerate(tasks):
                    row = pooled_out[idx]
                    logit, num_label = None, None
                    if task == 'oce':
                        gate = torch.sigmoid(self.gate1(row))
                        row = row * gate
                        logit = self.classifier_oce(row)
                        num_label = 7
                    elif task == 'ocnli':
                        gate = torch.sigmoid(self.gate2(row))
                        row = row * gate
                        logit = self.classifier_ocnli(row)
                        num_label = 3
                    elif task == 'tnews':
                        gate = torch.sigmoid(self.gate3(row))
                        row = row * gate
                        logit = self.classifier_tnews(row)
                        num_label = 15
                    logits.append(logit)
                    num_labels.append(num_label)
                outputs = (logits,)

                if labels is not None:
                    loss_l = []
                    for idx, label in enumerate(labels):
                        loss_l.append(self.loss_fct(logits[idx].view(-1, num_labels[idx]), label.view(-1)))
                    loss = torch.mean(torch.stack(loss_l))
                    outputs = (loss,) + outputs

                return outputs

# coding=utf-8
"""
@author: yyq90
@contact: yyq9012@gmail.com
@time: 2020/11/24
"""
import torch
from torch.utils.data import Dataset
from src_final.preprocess.processor import OceFeature,OcnliFeature,TnewsFeature

class BaseDataset(Dataset):
    def __init__(self, features, mode):
        self.nums = len(features)

        self.token_ids = [torch.tensor(example.token_ids).long() for example in features]
        self.attention_masks = [torch.tensor(example.attention_masks).float() for example in features]
        self.token_type_ids = [torch.tensor(example.token_type_ids).long() for example in features]

        self.labels = None
        if mode in ['train','dev']:
            self.labels = [torch.tensor(example.labels) for example in features]

    def __len__(self):
        return self.nums


class OceDataset(BaseDataset):
    def __init__(self,
                 features,
                 mode,
                 ):
        super(OceDataset, self).__init__(features, mode)

    def __getitem__(self, index):
        data = {'token_ids': self.token_ids[index],
                'attention_masks': self.attention_masks[index],
                'token_type_ids': self.token_type_ids[index]}

        if self.labels is not None:
            data['labels'] = self.labels[index]

        return data


class OcnliDataset(BaseDataset):
    def __init__(self,
                 features,
                 mode,
                 ):
        super(OcnliDataset, self).__init__(features, mode)

    def __getitem__(self, index):
        data = {'token_ids': self.token_ids[index],
                'attention_masks': self.attention_masks[index],
                'token_type_ids': self.token_type_ids[index]}

        if self.labels is not None:
            data['labels'] = self.labels[index]

        return data


class TnewsDataset(BaseDataset):
    def __init__(self,
                 features,
                 mode,
                 ):
        super(TnewsDataset, self).__init__(features, mode)

    def __getitem__(self, index):
        data = {'token_ids': self.token_ids[index],
                'attention_masks': self.attention_masks[index],
                'token_type_ids': self.token_type_ids[index]}

        if self.labels is not None:
            data['labels'] = self.labels[index]

        return data


class CombinDataset(BaseDataset):
    def __init__(self,
                 features,
                 mode,
                 ):
        super(CombinDataset,self).__init__(features,mode)
        self.tasks = []
        for feature in features:
            if isinstance(feature,OceFeature):
                self.tasks.append('oce')
            elif isinstance(feature,OcnliFeature):
                self.tasks.append('ocnli')
            elif isinstance(feature,TnewsFeature):
                self.tasks.append('tnews')


    def __getitem__(self, index):
        data = {'token_ids': self.token_ids[index],
                'attention_masks': self.attention_masks[index],
                'token_type_ids': self.token_type_ids[index],
                'task':self.tasks[index]}

        if self.labels is not None:
            data['labels'] = self.labels[index]

        return data

def build_dataset(task_type, features, mode, **kwargs):
    assert task_type in ['oce', 'tnews', 'ocnli','total'], 'task mismatch'

    if task_type == 'oce':
        dataset = OceDataset(features, mode)
    elif task_type == 'tnews':
        dataset = TnewsDataset(features, mode)
    elif task_type == 'ocnli':
        dataset = OcnliDataset(features, mode)
    elif task_type == 'total':
        dataset = CombinDataset(features,mode)

    return dataset

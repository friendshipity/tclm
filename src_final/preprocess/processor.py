# coding=utf-8
"""
@author: yyq90
@contact: yyq9012@gmail.com
@time: 2020/11/24
"""
import json
import random
import logging
from tqdm import tqdm
from transformers import BertTokenizer

logger = logging.getLogger(__name__)

__all__ = ['OceProcessor', 'OcnliProcessor', 'TnewsProcessor',
           'fine_grade_tokenize', 'search_label_index', 'convert_examples_to_features']


class BaseExample:
    def __init__(self,
                 set_type,
                 text,
                 label=None):
        self.set_type = set_type
        self.text = text
        self.label = label


class OceExample(BaseExample):
    def __init__(self,
                 set_type,
                 text,
                 label=None):
        super(OceExample, self).__init__(set_type=set_type,
                                         text=text,
                                         label=label)


class TnewsExample(BaseExample):
    def __init__(self,
                 set_type,
                 text,
                 label=None):
        super(TnewsExample, self).__init__(set_type=set_type,
                                           text=text,
                                           label=label)


class OcnliExample(BaseExample):
    def __init__(self,
                 set_type,
                 text_a,
                 text_b,
                 label=None):
        super(OcnliExample, self).__init__(set_type=set_type,
                                           text=text_a + '\t' + text_b,
                                           label=label)
        self.text_a = text_a
        self.text_b = text_b


class BaseFeature:
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 labels=None):
        self.token_ids = token_ids
        self.attention_masks = attention_masks
        self.token_type_ids = token_type_ids
        self.labels = labels


class OceFeature(BaseFeature):
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 labels=None):
        super(OceFeature, self).__init__(token_ids=token_ids,
                                         attention_masks=attention_masks,
                                         token_type_ids=token_type_ids,
                                         labels=labels)


class TnewsFeature(BaseFeature):
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 labels=None):
        super(TnewsFeature, self).__init__(token_ids=token_ids,
                                           attention_masks=attention_masks,
                                           token_type_ids=token_type_ids,
                                           labels=labels)


class OcnliFeature(BaseFeature):
    def __init__(self,
                 token_ids,
                 attention_masks,
                 token_type_ids,
                 labels=None):
        super(OcnliFeature, self).__init__(token_ids=token_ids,
                                           attention_masks=attention_masks,
                                           token_type_ids=token_type_ids,
                                           labels=labels)



class BaseProcessor:
    @staticmethod
    def read_json(file_path):
        with open(file_path, encoding='utf-8') as f:
            examples = json.load(f)
        return examples

    @staticmethod
    def load_raw(file_path):
        with open(file_path, encoding='utf-8') as f:
            examples = f.readlines()
        return examples


class OceProcessor(BaseProcessor):

    @staticmethod
    def _example_generator(raw_examples, set_type, label_dict=None):
        examples = []
        for line in raw_examples:
            line = line.replace('\n','')
            text = line.split('\t')[1]

            label_id = None
            if set_type != 'test':
                label = line.split('\t')[2]
                label_id = label_dict[label]

            examples.append(OceExample(set_type=set_type,
                                       text=text,
                                       label=label_id))
        return examples

    def get_labels(self):
        return ["fear", "happiness", "surprise", "sadness", "anger", "disgust", "like"]

    def get_labels_dict(self):
        return {label: i for i, label in enumerate(self.get_labels())}

    def get_train_examples(self, raw_examples):
        return self._example_generator(raw_examples, 'train', self.get_labels_dict())

    def get_dev_examples(self, raw_examples):
        return self._example_generator(raw_examples, 'dev', self.get_labels_dict())

    def get_test_examples(self, raw_examples):
        return self._example_generator(raw_examples, 'test')

class TnewsProcessor(BaseProcessor):

    @staticmethod
    def _example_generator(raw_examples, set_type, label_dict=None):
        examples = []
        for line in raw_examples:
            line = line.replace('\n','')

            text = line.split('\t')[1]
            label_id = None
            if set_type != 'test':
                label = line.split('\t')[2]
                label_id = label_dict[label]
            examples.append(TnewsExample(set_type=set_type,
                                       text=text,
                                       label=label_id))
        return examples

    def get_labels(self):
        return ["100", "101", "102", "103", "104", "106", "107", "108", "109", "110", "112", "113", "114", "115", "116"]

    def get_labels_dict(self):
        return {label: i for i, label in enumerate(self.get_labels())}

    def get_train_examples(self, raw_examples):
        return self._example_generator(raw_examples, 'train', self.get_labels_dict())

    def get_dev_examples(self, raw_examples):
        return self._example_generator(raw_examples, 'dev', self.get_labels_dict())

    def get_test_examples(self, raw_examples):
        return self._example_generator(raw_examples, 'test')


class OcnliProcessor(BaseProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def get_labels(self):
    #     """See base class."""
    #     return ["contradiction", "entailment", "neutral"]
    @staticmethod
    def _example_generator(raw_examples, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for line in raw_examples:
            line = line.replace('\n','')

            text_a = line.split('\t')[1]
            text_b = line.split('\t')[2]
            label_id = None
            if set_type != 'test':
                label = line.split('\t')[3]
                label_id = int(label)
            examples.append(OcnliExample(set_type=set_type,
                                         text_a=text_a,
                                         text_b=text_b,
                                         label=label_id))
        return examples

    def get_train_examples(self, raw_examples):
        return self._example_generator(raw_examples, 'train')

    def get_dev_examples(self, raw_examples):
        return self._example_generator(raw_examples, 'dev')

    def get_test_examples(self, raw_examples):
        return self._example_generator(raw_examples, 'test')






def search_label_index(tokens, label_tokens):
    """
    search label token indexes in all tokens
    :param tokens: tokens for raw text
    :param label_tokens: label which are split by the cjk extractor
    :return:
    """
    index_list = []  # 存放搜到的所有的index

    # 滑动窗口搜索 labels 在 token 中的位置
    for index in range(len(tokens) - len(label_tokens) + 1):
        if tokens[index: index + len(label_tokens)] == label_tokens:
            start_index = index
            end_index = start_index + len(label_tokens) - 1
            index_list.append((start_index, end_index))

    return index_list


def fine_grade_tokenize(raw_text, tokenizer):
    """
    序列标注任务 BERT 分词器可能会导致标注偏移，
    用 char-level 来 tokenize
    """
    tokens = []

    for _ch in raw_text:
        if _ch in [' ', '\t', '\n']:
            # tokens.append('[BLANK]')
            pass
        else:
            if not len(tokenizer.tokenize(_ch)):
                tokens.append('[INV]')
            else:
                tokens.append(_ch)

    return tokens


def convert_oce_example(ex_idx, example: OceExample, max_seq_len, tokenizer: BertTokenizer):
    """
    convert oce emotion examples to trigger features
    """
    set_type = example.set_type
    raw_text = example.text
    raw_label = example.label

    tokens = fine_grade_tokenize(raw_text, tokenizer)

    labels = raw_label

    encode_dict = tokenizer.encode_plus(text=tokens,
                                        max_length=max_seq_len,
                                        pad_to_max_length=True,
                                        is_pretokenized=True,
                                        return_token_type_ids=True,
                                        return_attention_mask=True)

    token_ids = encode_dict['input_ids']
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']

    if ex_idx < 3 and set_type == 'train':
        logger.info(f"*** {set_type}_example-{ex_idx} ***")
        logger.info(f'text: {" ".join(tokens)}')
        logger.info(f"token_ids: {token_ids}")
        logger.info(f"attention_masks: {attention_masks}")
        logger.info(f"token_type_ids: {token_type_ids}")

    feature = OceFeature(token_ids=token_ids,
                         attention_masks=attention_masks,
                         token_type_ids=token_type_ids,
                         labels=labels)

    return feature


def convert_tnews_example(ex_idx, example: TnewsExample, max_seq_len, tokenizer: BertTokenizer):
    """
    convert trigger examples to trigger features
    """
    set_type = example.set_type
    raw_text = example.text
    raw_label = example.label

    tokens = fine_grade_tokenize(raw_text, tokenizer)

    labels = raw_label

    encode_dict = tokenizer.encode_plus(text=tokens,
                                        max_length=max_seq_len,
                                        pad_to_max_length=True,
                                        is_pretokenized=True,
                                        return_token_type_ids=True,
                                        return_attention_mask=True)

    token_ids = encode_dict['input_ids']
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']

    if ex_idx < 3 and set_type == 'train':
        logger.info(f"*** {set_type}_example-{ex_idx} ***")
        logger.info(f'text: {" ".join(tokens)}')
        logger.info(f"token_ids: {token_ids}")
        logger.info(f"attention_masks: {attention_masks}")
        logger.info(f"token_type_ids: {token_type_ids}")

    feature = TnewsFeature(token_ids=token_ids,
                           attention_masks=attention_masks,
                           token_type_ids=token_type_ids,
                           labels=labels)

    return feature


def convert_ocnli_example(ex_idx, example: OcnliExample, max_seq_len, tokenizer: BertTokenizer):
    """
    convert trigger examples to trigger features
    """
    set_type = example.set_type
    text_a = example.text_a
    text_b = example.text_b
    raw_label = example.label

    tokens_a = fine_grade_tokenize(text_a, tokenizer)
    tokens_b = fine_grade_tokenize(text_b, tokenizer)

    labels = raw_label

    encode_dict = tokenizer.encode_plus(text=tokens_a,
                                        text_pair=tokens_b,
                                        max_length=max_seq_len,
                                        pad_to_max_length=True,
                                        is_pretokenized=True,
                                        return_token_type_ids=True,
                                        return_attention_mask=True)

    token_ids = encode_dict['input_ids']
    attention_masks = encode_dict['attention_mask']
    token_type_ids = encode_dict['token_type_ids']

    if ex_idx < 3 and set_type == 'train':
        logger.info(f"*** {set_type}_example-{ex_idx} ***")
        logger.info(f'text_a: {" ".join(tokens_a)}')
        logger.info(f"token_ids: {token_ids}")
        logger.info(f"attention_masks: {attention_masks}")
        logger.info(f"token_type_ids: {token_type_ids}")

    feature = OcnliFeature(token_ids=token_ids,
                           attention_masks=attention_masks,
                           token_type_ids=token_type_ids,
                           labels=labels)

    return feature


def convert_examples_to_features(task_type, examples, bert_dir, max_seq_len, **kwargs):
    assert task_type in ['oce', 'tnews', 'ocnli','total']

    tokenizer = BertTokenizer.from_pretrained(bert_dir)
    logger.info(f'Vocab nums in this tokenizer is: {tokenizer.vocab_size}')

    features = []

    for i, example in enumerate(tqdm(examples, desc=f'convert examples')):
        if task_type == 'oce':

            feature = convert_oce_example(
                ex_idx=i,
                example=example,
                max_seq_len=max_seq_len,
                tokenizer=tokenizer,
            )

        elif task_type == 'tnews':
            feature = convert_tnews_example(
                ex_idx=i,
                example=example,
                max_seq_len=max_seq_len,
                tokenizer=tokenizer
            )

        elif task_type == 'ocnli':
            feature = convert_ocnli_example(
                ex_idx=i,
                example=example,
                max_seq_len=max_seq_len,
                tokenizer=tokenizer
            )

        elif task_type == 'total':
            if isinstance(example,OceExample):
                feature = convert_oce_example(
                    ex_idx=i,
                    example=example,
                    max_seq_len=max_seq_len,
                    tokenizer=tokenizer,
                )
            elif isinstance(example,OcnliExample):
                feature = convert_ocnli_example(
                    ex_idx=i,
                    example=example,
                    max_seq_len=max_seq_len,
                    tokenizer=tokenizer
                )
            elif isinstance(example,TnewsExample):
                feature = convert_tnews_example(
                    ex_idx=i,
                    example=example,
                    max_seq_len=max_seq_len,
                    tokenizer=tokenizer
                )
        features.append(feature)

    return features

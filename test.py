# coding=utf-8

import os
import logging
import torch
import numpy as np
import tqdm
import ipdb

from torch.utils.data import DataLoader
from src_final.preprocess.processor import *
from src_final.utils.trainer import train
from src_final.utils.options import TrainArgs
from src_final.utils.model_utils import bertForSequenceClassification as build_model
from src_final.utils.dataset_utils import build_dataset
from src_final.utils.functions_utils import set_seed, get_model_path_list, load_model_and_parallel
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report ,f1_score

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

def get_base_out(task, model, loader, device):
    model.eval()
    with torch.no_grad():
        for idx, _batch in enumerate(loader):
            for key in _batch.keys():
                _batch[key] = _batch[key].to(device)
                tmp_out = model(task=task,**_batch)
                yield tmp_out[1],_batch['labels']

def eval(opt):
    model_path = opt.model_path
    processors = {'oce': (OceProcessor, 'OCEMOTION'),
                  'tnews': (TnewsProcessor, 'TNEWS'),
                  'ocnli': (OcnliProcessor, 'OCNLI')
                  }
    model = build_model(opt.bert_dir)
    for task in processors.keys():
        logger.info(
            f'------------------------------------------------------------Task : {task} eval ---------------------------------------------------------------')
        processor = processors[task][0]()
        dev_examples = None
        file_dev = processors[task][1] + '_dev_s.csv'
        dev_raw_examples = processor.load_raw(os.path.join(opt.raw_data_dir, file_dev))
        dev_examples = processor.get_dev_examples(dev_raw_examples)
        dev_features = convert_examples_to_features(task, dev_examples, opt.bert_dir, opt.max_seq_len)
        dev_dataset = build_dataset(task, dev_features, 'dev')

        dev_loader = DataLoader(dev_dataset, batch_size=opt.eval_batch_size, shuffle=False, num_workers=0)


        model, device = load_model_and_parallel(model, opt.gpu_ids[0], ckpt_path=model_path)
        pred_logits = None
        labels = None
        for tmp_pred in get_base_out(task, model, dev_loader, device):
            pred = tmp_pred[0].cpu().numpy()
            label = tmp_pred[1].cpu().numpy()

            if pred_logits is None:
                pred_logits = pred
            else:
                pred_logits = np.append(pred_logits, pred, axis=0)

            if labels is None:
                labels = label
            else:
                labels = np.append(labels, label, axis=0)

        preds = np.argmax(pred_logits, -1)
        macro_f1_score = f1_score(labels, preds, average='macro')
        logger.info(f'\nmacro f1 at {model_path} is {macro_f1_score}')
        logger.info(f'\nconfusion_matrix:')
        cm_res = confusion_matrix(labels, preds, )
        logger.info(f'{cm_res}')
        logger.info(f'{classification_report(labels,preds,)}')


def training(opt):
    processors = {'oce': (OceProcessor,'OCEMOTION'),
                  'tnews': (TnewsProcessor,'TNEWS'),
                  'ocnli': (OcnliProcessor,'OCNLI')
                  }
    devs = []
    model = build_model(opt.bert_dir)
    out_dir_base = opt.output_dir
    for task in processors.keys():
        logger.info(f'------------------------------------------------------------Task : {task} trainnig ---------------------------------------------------------------')
        processor = processors[task][0]()
        file_train = processors[task][1]+'_train_s.csv'
        train_raw_examples = processor.load_raw(os.path.join(opt.raw_data_dir, file_train))
        train_examples = processor.get_train_examples(train_raw_examples)
        train_features = convert_examples_to_features(task, train_examples, opt.bert_dir,
                                                      opt.max_seq_len)
        logger.info(f'Build {len(train_features)} train features')
        train_dataset = build_dataset(task, train_features, 'train')

        opt.output_dir = os.path.join(out_dir_base,opt.bert_type,task)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
        train(task, opt, model, train_dataset)

        if opt.eval_model:
            file_dev = processors[task][1] + '_dev_s.csv'
            dev_raw_examples = processor.load_raw(os.path.join(opt.raw_data_dir, file_dev))
            dev_examples = processor.get_dev_examples(dev_raw_examples)
            dev_features = convert_examples_to_features(task, dev_examples, opt.bert_dir, opt.max_seq_len)
            dev_dataset = build_dataset(task, dev_features, 'dev')
            dev_loader = DataLoader(dev_dataset, batch_size=opt.eval_batch_size,shuffle=False,num_workers=0)
            model_path_list = get_model_path_list(opt.output_dir)

            for idx, model_path in enumerate(model_path_list):
                tmp_step = model_path.split('/')[-2].split('-')[-1]
                model,device = load_model_and_parallel(model, opt.gpu_ids[0], ckpt_path=model_path)
                pred_logits = None
                labels = None
                for tmp_pred in get_base_out(task, model, dev_loader, device):
                    pred = tmp_pred[0].cpu().numpy()
                    label = tmp_pred[1].cpu().numpy()

                    if pred_logits is None:
                        pred_logits = pred
                    else:
                        pred_logits = np.append(pred_logits, pred, axis=0)

                    if labels is None:
                        labels = label
                    else:
                        labels = np.append(labels, label, axis=0)

                preds = np.argmax(pred_logits,-1)
                macro_f1_score = f1_score(labels,preds,average='macro')
                logger.info(f'\nmacro f1 at {model_path} is {macro_f1_score}')
                logger.info(f'\nconfusion_matrix:')
                cm_res = confusion_matrix(labels, preds,)
                logger.info(f'{cm_res}')
                logger.info(f'{classification_report(labels,preds,)}')

if __name__ == '__main__':
    args = TrainArgs().get_parser()
    set_seed(seed=123)

    if args.attack_train != '':
        args.output_dir += f'_{args.attack_train}'

    if args.weight_decay:
        args.output_dir += '_wd'



    if args.mode == 'train':
        training(args)

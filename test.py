# coding=utf-8

import os
import logging
import torch
import numpy as np
import tqdm
import ipdb
import json
from torch.utils.data import DataLoader
from src_final.preprocess.processor import *
from src_final.utils.options import TrainArgs
from src_final.utils.model_utils import bertForSequenceClassification as build_model
# from src_final.utils.model_utils import GateOut as build_model
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

            yield tmp_out[0]



def training(opt):
    processors = {'oce': (OceProcessor,'OCEMOTION'),
                  'tnews': (TnewsProcessor,'TNEWS'),
                  'ocnli': (OcnliProcessor,'OCNLI')
                  }
    model = build_model(opt.bert_dir)
    for task in processors.keys():
        logger.info(
            f'------------------------------------------------------------Task : {task} load ---------------------------------------------------------------')

        file_test = processors[task][1] + '_a.csv'
        processor = processors[task][0]()
        label_dict = None
        if task in ['oce','tnews']:
            label_dict = processor.get_labels()



        test_raw_examples = processor.load_raw(os.path.join(opt.raw_data_dir, file_test))
        test_examples = processor.get_test_examples(test_raw_examples)
        test_features = convert_examples_to_features(task, test_examples, opt.bert_dir, opt.max_seq_len)
        _dataset = build_dataset(task, test_features, 'test')
        test_loader = DataLoader(_dataset, batch_size=opt.eval_batch_size,shuffle=False,num_workers=0)

        ckpt_path = None
        if opt.ckpt_path:
            ckpt_path = opt.ckpt_path
        model,device = load_model_and_parallel(model, opt.gpu_ids[0], ckpt_path=ckpt_path)
        pred_logits = None
        labels = None
        for tmp_pred in get_base_out(task, model, test_loader, device):
            pred = tmp_pred.cpu().numpy()

            if pred_logits is None:
                pred_logits = pred
            else:
                pred_logits = np.append(pred_logits, pred, axis=0)

        pred_idx = np.argmax(pred_logits,-1)
        if label_dict:
            preds = [label_dict[i] for i in pred_idx]
        else:
            preds = [str(i) for i in pred_idx]

        js_l = []
        for i in range(len(preds)):
            js_l.append({"id":str(i),'label':preds[i]})
        if not os.path.exists('./submits'):
            os.makedirs('./submits', exist_ok=True)
        with open('./submits/'+processors[task][1].lower()+'_predict.json','w') as f:
            for js in js_l:
                json.dump(js,f)
                f.write('\n')
        f.close()
        print()



if __name__ == '__main__':
    args = TrainArgs().get_parser()
    set_seed(seed=123)

    training(args)

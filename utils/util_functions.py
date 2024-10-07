import random
import warnings
import time
import logging
import yaml
import os
import importlib
import torch
import numpy as np
from typing import Callable, Any
from easydict import EasyDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef

import torch.nn.functional as F

from .util_classes import MyPrint

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def prepare(config):
    # set TRAIN_PATH
    TRAIN_PATH = f'./runs/{int(time.time())}'
    if not os.path.exists(TRAIN_PATH):
        os.makedirs(TRAIN_PATH)
    warnings.filterwarnings("ignore")
    # set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'{TRAIN_PATH}/run.log')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # set device
    device = torch.device(f'cuda:{config.train.gpu_id}' if torch.cuda.is_available() else 'cpu')
    # set print
    my_print = MyPrint(logger).pprint
    my_print('TRAIN_PATH:', TRAIN_PATH)
    my_print('Now the time is:', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    my_print(config)
    config.model.esm_path = os.path.join(config.model.model_dir, config.model.esm_version)
    config.train.save_path = TRAIN_PATH
    return logger, device, my_print


def read_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as r:
        config = EasyDict(yaml.safe_load(r))
    set_seed(config.settings.seed)
    return config


def get_model(pprint, config):
    module = importlib.import_module('model')
    model_class = getattr(module, config.model_name)
    model = model_class(config=config)
    pprint('total params:', sum(p.numel() for p in model.parameters()))
    pprint('trainable params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


def find_continuous_ones(arr):
    continuous_ones = []
    start = None
    for i, l in enumerate(arr):
        if l == 1 and start is None:
            start = i
        elif l == 0 and start is not None:
            continuous_ones.append((start, i))
            start = None
    if start is not None:
        continuous_ones.append((start, len(arr)))
    return continuous_ones

def cal_os(label, pred):
    conti_ones_label = find_continuous_ones(label)
    conti_ones_pred = find_continuous_ones(pred)
    fenzi, fenmu = 0, 0
    for i in range(len(conti_ones_pred)):
        for j in range(len(conti_ones_label)):
            l = min(conti_ones_pred[i][1], conti_ones_label[j][1])
            r = max(conti_ones_pred[i][0], conti_ones_label[j][0])
            fenzi += max(0, l - r)
    if len(conti_ones_label) == 0:
        return 1
    for i in range(len(conti_ones_label)):
        fenmu += conti_ones_label[i][1] - conti_ones_label[i][0]
    return fenzi / fenmu

def intersect(a1, b1, a2, b2):
    if a2 >= b1 or a1 >= b2:
        return False
    return True

def cal_fpr(label, pred):
    conti_ones_label = find_continuous_ones(label)
    conti_ones_pred = find_continuous_ones(pred)
    fenzi, fenmu = 0, 0
    if len(conti_ones_pred) == 0:
        return 1
    for i in range(len(conti_ones_pred)):
        to_add = conti_ones_pred[i][1] - conti_ones_pred[i][0]
        flag = 0
        for j in range(len(conti_ones_label)):
            if intersect(conti_ones_pred[i][0], conti_ones_pred[i][1], conti_ones_label[j][0], conti_ones_label[j][1]):
                flag = 1
                break
        if flag == 0:
            fenzi += to_add
        fenmu += to_add
    return fenzi / fenmu

def measure(all_labels, all_preds, probabilities, length):

    accuracy, precision, recall, f1, auc, prc, mcc, oss, fpr = 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    labels = [[label for label in sublist[1:length+1]] for sublist, length in zip(all_labels, length)]
    preds = [[pred for pred in sublist[1:length+1]] for sublist, length in zip(all_preds, length)]
    ps = [[p for p in sublist[1:length+1]] for sublist, length in zip(probabilities, length)]
    
    for idx in range(len(labels)):
        label, pred, p = labels[idx], preds[idx], ps[idx]
        accuracy += accuracy_score(label, pred)
        precision += precision_score(label, pred, average='binary')
        recall += recall_score(label, pred, average='binary')
        f1 += f1_score(label, pred, average='binary')
        auc += roc_auc_score(label, p, average='macro')
        prc += average_precision_score(label, p, average='macro')
        mcc += matthews_corrcoef(label, pred)
        oss += cal_os(label, pred)
        fpr += cal_fpr(label, pred)
    
    return {
        'accuracy': accuracy / len(labels),
        'precision': precision / len(labels),
        'recall': recall / len(labels),
        'f1': f1 / len(labels),
        'auc': auc / len(labels),
        'prc': prc / len(labels),
        'mcc': mcc / len(labels),
        'oss': oss / len(labels),
        'fpr': fpr / len(labels)
    }


def log_metrics(pprint: Callable[..., Any], thistype: str, epoch: int = None, metric_dict: dict = None, annot: str = None) -> None:
    if thistype == 'train':
        train_or_val = 'Train'
    elif thistype == 'val':
        train_or_val = 'Val'
    else:
        train_or_val = 'Test'
    info_str = f'{train_or_val} Epoch: {epoch} - {annot} : ' if epoch is not None else f'{train_or_val}: '
    for metric in metric_dict:
        info_str += f'{metric}: {metric_dict[metric]:.8f}, '
    pprint(info_str)


def cos_sim(a, b):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def kl_loss(logits, softlabel, tau=0.5, way='mean', use_loss="kl"):
    sim_targets = F.softmax(softlabel / tau, dim=1)
    logit_inputs = F.log_softmax(logits / tau, dim=1)

    if use_loss == "kl":
        loss = F.kl_div(logit_inputs, sim_targets, reduction='batchmean')
    elif use_loss == "contrastive":
        if way == 'mean':
            loss = -torch.mean(logit_inputs * sim_targets, dim=1).mean()
        elif way == 'sum':
            loss = -torch.sum(logit_inputs * sim_targets, dim=1).mean()
    return loss
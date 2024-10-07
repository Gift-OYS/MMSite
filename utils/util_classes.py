import pandas as pd
import torch
import json
import math
from transformers import AutoTokenizer

from . import get_max_length, get_str, get_col_name_list


class MyPrint():
    def __init__(self, logger):
        self.logger = logger
        
    def pprint(self, *args):
        print(*args)
        log_message = ', '.join(str(arg) for arg in args)
        self.logger.info(log_message)


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_ratio=0.1, last_epoch=-1):
        self.total_steps = total_steps
        self.warmup_steps = total_steps * warmup_ratio
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            lr_scale = (self.last_epoch + 1) / self.warmup_steps
            return [base_lr * lr_scale for base_lr in self.base_lrs]
        else:
            cosine_step = self.last_epoch - self.warmup_steps
            cosine_steps = self.total_steps - self.warmup_steps
            return [
                base_lr * (1 + math.cos(math.pi * cosine_step / cosine_steps)) / 2
                for base_lr in self.base_lrs
            ]


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, config, type=None, align=False):
        self.config = config
        self.type = type
        self.max_length = config.dataset.max_length
        self.align = align
        self.function_len = 128

        self.data = pd.read_csv(f'{config.dataset.data_path}/{type}.tsv', sep='\t')
        if type != 'infer':
            self.data = self._filter_min_attr_num(self.data)
            self._drop_unused_prots()
        
        self.seq_tokenizer = AutoTokenizer.from_pretrained(f'{config.model.model_dir}/{config.model.esm_version}')
        self.text_tokenizer = AutoTokenizer.from_pretrained(f'{config.model.model_dir}/{config.model.pubmed_version}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data['Entry'][idx]
        sequence = self.data['Sequence'][idx]
        actual_length = min(len(sequence), self.max_length-2)

        if self.type != 'infer':
            active_sites = self.data['Active site'][idx]
            as_label = torch.zeros(self.max_length)
            if type(active_sites) == str:
                for active_site in active_sites.split(','):
                    if int(active_site) < self.max_length - 1:
                        as_label[int(active_site)] = 1.0

        seq = self._tokenization(sequence, 'sequence')
        
        if self.type == 'train' or self.align:
            texts = []
            for col in get_col_name_list():
                text = get_str(self.data, col, idx)
                max_length = get_max_length(col, self.config)
                encoded_text = self._tokenization(text, 'text', max_length)
                texts.append(encoded_text)
        else:
            prior_knowledge = json.load(open(self.config.prior_knowledge, 'r'))
            texts = prior_knowledge[entry]
            texts = self._tokenization(texts, 'text', 128)

        if self.type != 'infer':
            return texts, seq, as_label, actual_length, {'Entry': entry, 'Active sites': active_sites}
        else:
            return texts, seq, actual_length, {'Entry': entry}

    def _filter_min_attr_num(self, data, min_attr_num=6):
        cols = get_col_name_list()
        cols.remove('Function')
        data = data.reset_index(drop=True)
        
        for idx in range(len(data)):
            if type(data['Function'][idx]) != str:
                data = data.drop(idx)
                continue
            cnt = 0
            for col in cols:
                if type(data[col][idx]) == str:
                    cnt += 1
            if cnt < min_attr_num - 1:
                data = data.drop(idx)
        data = data.reset_index(drop=True)
        return data

    def _drop_unused_prots(self):
        self.data = self.data.reset_index(drop=True)
        for idx in range(len(self.data)):
            active_sites = self.data['Active site'][idx]
            max_as = max([int(as_) for as_ in active_sites.split(',')])
            if max_as >= self.max_length - 1:
                self.data = self.data.drop(idx)
        self.data = self.data.reset_index(drop=True)

    def _tokenization(self, sample, modal, max_length=None):
        if modal == 'text':
            tokenizer = self.text_tokenizer
        else:
            tokenizer = self.seq_tokenizer
        max_length = max_length if max_length is not None else self.max_length
        
        encoded_sample = tokenizer(sample, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")
        encoded_sample['input_ids'] = torch.squeeze(encoded_sample['input_ids'])
        encoded_sample['attention_mask'] = torch.squeeze(encoded_sample['attention_mask'])
        return encoded_sample

import torch
import os
from torch import optim
from tqdm import tqdm
import torch.nn as nn
import argparse
import sys
proj_path = os.path.abspath('.')
sys.path.append(proj_path)

from utils.util_functions import read_config, log_metrics, measure, prepare, get_model
from utils.util_classes import MyDataset, WarmupCosineAnnealingLR
from torch.utils.data import DataLoader

device, pprint = None, None

def align(model, train_loader, align_valid_loader):
    epochs = config.train.align_epoch
    best_f_score_cl, best_metric_cl, info_str_cl = 100000, {}, 'Best val scores (Contrastive Loss): '

    bce_loss = nn.BCELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.train.lr)
    scheduler = WarmupCosineAnnealingLR(optimizer, total_steps=epochs*len(train_loader))
    model.to(device)

    for epoch in range(epochs):
        model.train()
        tr_metric_dict_cl = {}
        for metric in config.train.metrics:
            tr_metric_dict_cl[metric] = 0.0

        for batch in tqdm(train_loader, desc=f"Training: Epoch {epoch+1}/{epochs}"):
            anchor_text_input_ids = [b['input_ids'].to(device) for b in batch[0]]
            anchor_text_attention_mask = [b['attention_mask'].to(device) for b in batch[0]]
            anchor_seq_input_ids = batch[1]['input_ids'].to(device)
            anchor_seq_attention_mask = batch[1]['attention_mask'].to(device)

            optimizer.zero_grad()
            output = model(anchor_text_input_ids=anchor_text_input_ids, anchor_text_attention_mask=anchor_text_attention_mask,
                            anchor_seq_input_ids=anchor_seq_input_ids, anchor_seq_attention_mask=anchor_seq_attention_mask)
            loss = output['contrastive_loss'].float()
            loss.backward()
            optimizer.step()
            scheduler.step()
            tr_metric_dict_cl['loss'] += loss.item()

        for k in tr_metric_dict_cl.keys():
            tr_metric_dict_cl[k] /= len(train_loader)

        log_metrics(pprint, 'train', epoch+1, tr_metric_dict_cl, 'Contrastive Loss')
        _, val_metric_dict_cl = evaluate(epochs, epoch, model, align_valid_loader, bce_loss, align=True)

        if epoch == 0 or val_metric_dict_cl['loss'] < best_f_score_cl:
            best_f_score_cl = val_metric_dict_cl['loss']
            best_metric_cl = val_metric_dict_cl
            
            exist_files = [f for f in os.listdir(config.train.save_path) if f'best_model_align' in f]
            for f in exist_files:
                os.remove(os.path.join(config.train.save_path, f))
            checkpoint_align = f"{config.train.save_path}/best_model_align_{best_f_score_cl}.pth"
            pprint(f"Saving the best model to {checkpoint_align}")
            torch.save(model, checkpoint_align)

    for k, v in best_metric_cl.items():
        info_str_cl += f"{k}: {v:.4f}, "
    pprint(info_str_cl)
    
    return checkpoint_align
    

def fuse(checkpoint_align, train_loader, val_loader):
    model = torch.load(checkpoint_align)
    model.train()
    model.to(device)
    suffixes = ['seq_suffix_encoder', 'seq_suffix_transformer', 'token_suffix_encoder_res', 'token_suffix_transformer_res', 'token_suffix_encoder', 
                'token_suffix_transformer', 'fusion_cross', 'bn2_res', 'fc2_res', 'bn2', 'fc2', 'classifier_token']
    for k, v in model.named_parameters():
        v.requires_grad = any(suffix in k for suffix in suffixes)
    pprint('trainable params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    epochs=config.train.epochs
    bce_loss = nn.BCELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.train.lr)
    scheduler = WarmupCosineAnnealingLR(optimizer, total_steps=epochs*len(train_loader))
    
    best_f_score_token, best_metric_token, info_str_token = 0.0, {}, 'Best val scores (Token): '

    for epoch in range(epochs):
        model.train()
        tr_metric_dict_token = {}
        for metric in config.train.metrics:
            tr_metric_dict_token[metric] = 0.0

        for batch in tqdm(train_loader, desc=f"Training: Epoch {epoch+1}/{epochs}"):
            anchor_text_input_ids = [b['input_ids'].to(device) for b in batch[0]]
            anchor_text_attention_mask = [b['attention_mask'].to(device) for b in batch[0]]
            anchor_seq_input_ids = batch[1]['input_ids'].to(device)
            anchor_seq_attention_mask = batch[1]['attention_mask'].to(device)
            token_labels = batch[2].to(device)
            length = batch[3]

            optimizer.zero_grad()
            output = model(anchor_text_input_ids=anchor_text_input_ids, anchor_text_attention_mask=anchor_text_attention_mask,
                            anchor_seq_input_ids=anchor_seq_input_ids, anchor_seq_attention_mask=anchor_seq_attention_mask)
            token_loss = bce_loss(output['token_logits'].float(), token_labels.float())
            token_loss.backward()
            optimizer.step()
            scheduler.step()
            tr_metric_dict_token['loss'] += token_loss.item()
            token_predictions = (output['token_logits'] > config.metric.token_threshold).float()
            token_measurements = measure(token_labels.cpu().numpy(), token_predictions.cpu().numpy(), output['token_logits'].detach().to(torch.float).cpu().numpy(), length)
            for k, v in token_measurements.items():
                tr_metric_dict_token[k] += v

        for k in tr_metric_dict_token.keys():
            tr_metric_dict_token[k] /= len(train_loader)

        log_metrics(pprint, 'train', epoch+1, tr_metric_dict_token, 'Token')
        val_metric_dict_token, _ = evaluate(epochs, epoch, model, val_loader, bce_loss, align=False)

        if epoch == 0 or val_metric_dict_token['f1'] > best_f_score_token:
            best_f_score_token = val_metric_dict_token['f1']
            best_metric_token = val_metric_dict_token
            
            exist_files = [f for f in os.listdir(config.train.save_path) if f'best_model_fuse' in f]
            for f in exist_files:
                os.remove(os.path.join(config.train.save_path, f))
            checkpoint_path_total = f"{config.train.save_path}/best_model_fuse_{best_f_score_token}.pth"
            pprint(f"Saving the best model to {checkpoint_path_total}")
            torch.save(model, checkpoint_path_total)
    
    for k, v in best_metric_token.items():
        info_str_token += f"{k}: {v:.4f}, "
    pprint(info_str_token)


def evaluate(epochs, epoch, model, val_loader, bce_loss, align=False):
    model.eval()
    val_metric_dict_token, val_metric_dict_cl = {}, {}
    for metric in config.train.metrics:
        val_metric_dict_token[metric], val_metric_dict_cl[metric] = 0.0, 0.0
    acc_dataloader_size_token, acc_dataloader_size_cl = len(val_loader), len(val_loader)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation: Epoch {epoch+1}/{epochs}"):
            if align:
                anchor_text_input_ids = [b['input_ids'].to(device) for b in batch[0]]
                anchor_text_attention_mask = [b['attention_mask'].to(device) for b in batch[0]]
            else:
                anchor_text_input_ids = batch[0]['input_ids'].to(device)
                anchor_text_attention_mask = batch[0]['attention_mask'].to(device)
            anchor_seq_input_ids = batch[1]['input_ids'].to(device)
            anchor_seq_attention_mask = batch[1]['attention_mask'].to(device)
            token_labels = batch[2].to(device)
            length = batch[3]

            output = model(anchor_text_input_ids=anchor_text_input_ids, anchor_text_attention_mask=anchor_text_attention_mask,
                           anchor_seq_input_ids=anchor_seq_input_ids, anchor_seq_attention_mask=anchor_seq_attention_mask, test=not align)
            token_loss = bce_loss(output['token_logits'].float(), token_labels.float())
            contrastive_loss = output['contrastive_loss'].float()
            val_metric_dict_token['loss'] += token_loss.item()
            val_metric_dict_cl['loss'] += contrastive_loss.item()
            token_predictions = (output['token_logits'] > config.metric.token_threshold).float()
            token_measurements = measure(token_labels.cpu().numpy(), token_predictions.cpu().numpy(), output['token_logits'].detach().to(torch.float).cpu().numpy(), length)
            for k, v in token_measurements.items():
                val_metric_dict_token[k] += v

    for k in val_metric_dict_token.keys():
        val_metric_dict_token[k] /= acc_dataloader_size_token
    for k in val_metric_dict_cl.keys():
        val_metric_dict_cl[k] /= acc_dataloader_size_cl

    log_metrics(pprint, 'val', epoch+1, val_metric_dict_token, 'Token')
    log_metrics(pprint, 'val', epoch+1, val_metric_dict_cl, 'Contrastive Loss')
    
    return val_metric_dict_token, val_metric_dict_cl


def test(checkpoint_path, test_loader):
    model = torch.load(checkpoint_path, map_location=device)
    model.eval()
    test_metric_dict_token = {}
    for metric in config.train.metrics:
        test_metric_dict_token[metric] = 0.0
    acc_dataloader_size_token = len(test_loader)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing ..."):
            anchor_text_input_ids = batch[0]['input_ids'].to(device)
            anchor_text_attention_mask = batch[0]['attention_mask'].to(device)
            anchor_seq_input_ids = batch[1]['input_ids'].to(device)
            anchor_seq_attention_mask = batch[1]['attention_mask'].to(device)
            token_labels = batch[2].to(device)
            length = batch[3]
            
            output = model(anchor_text_input_ids=anchor_text_input_ids, anchor_text_attention_mask=anchor_text_attention_mask,
                            anchor_seq_input_ids=anchor_seq_input_ids, anchor_seq_attention_mask=anchor_seq_attention_mask, test=True)
            token_predictions = (output['token_logits'] > config.metric.token_threshold).float()
            token_measurements = measure(token_labels.cpu().numpy(), token_predictions.cpu().numpy(), output['token_logits'].detach().to(torch.float).cpu().numpy(), length)
            if not token_measurements:
                acc_dataloader_size_token -= 1
            else:
                for k, v in token_measurements.items():
                    test_metric_dict_token[k] += v

    for k in test_metric_dict_token.keys():
        test_metric_dict_token[k] /= acc_dataloader_size_token

    log_metrics(pprint, 'test', None, test_metric_dict_token, 'Token')


def prepare_dataloaders(config):
    train_dataset = MyDataset(config, 'train')
    valid_dataset = MyDataset(config, 'valid')
    align_val_dataset = MyDataset(config, 'valid', align=True)
    test_dataset = MyDataset(config, 'test')

    train_dataloader = DataLoader(train_dataset, batch_size=config.train.batch_size, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.val.batch_size, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.test.batch_size, drop_last=True)
    align_valid_loader = DataLoader(align_val_dataset, batch_size=config.val.batch_size, drop_last=True)
    return train_dataloader, valid_dataloader, align_valid_loader, test_dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help="running configurations", type=str, required=False, default=f'{proj_path}/configs/config.yaml')
    args = parser.parse_args()
    config = read_config(args.config)
    _, device, pprint = prepare(config)

    train_loader, val_loader, align_valid_loader, test_loader = prepare_dataloaders(config)
    model = get_model(pprint, config)
    
    pprint("Aligning ...")
    checkpoint_align = align(model, train_loader, align_valid_loader)
    pprint("Fusing ...")
    checkpoint_path_total = fuse(checkpoint_align, train_loader, val_loader)
    pprint("Testing ...")
    test(checkpoint_path_total, test_loader)

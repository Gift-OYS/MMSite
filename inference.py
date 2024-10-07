import os
import torch
import torch.nn as nn
import sys
proj_path = os.path.abspath('.')
sys.path.append(proj_path)

from utils.util_functions import read_config, prepare, get_model
from utils.util_classes import MyDataset
from torch.utils.data import DataLoader

writer, device, pprint = None, None, None
token_threshold = 0.5

config = read_config('/path/to/config.yaml')
_, device, pprint = prepare(config)

model = get_model(pprint, config)
bce_loss = nn.BCELoss()
checkpoint_path_total = '/path/to/runs/timestamp/best_model_fuse_xxx.pth'
model = torch.load(checkpoint_path_total, map_location=device)
model.eval()

test_dataset = MyDataset(config, 'infer')
test_loader = DataLoader(test_dataset, batch_size=24, drop_last=False)

with torch.no_grad():
    for batch in test_loader:
        anchor_text_input_ids = batch[0]['input_ids'].to(device)
        anchor_text_attention_mask = batch[0]['attention_mask'].to(device)
        anchor_seq_input_ids = batch[1]['input_ids'].to(device)
        anchor_seq_attention_mask = batch[1]['attention_mask'].to(device)
        length = batch[3]
        
        output = model(anchor_text_input_ids=anchor_text_input_ids, anchor_text_attention_mask=anchor_text_attention_mask,
                        anchor_seq_input_ids=anchor_seq_input_ids, anchor_seq_attention_mask=anchor_seq_attention_mask, test=True)
        token_predictions = (output['token_logits'] > token_threshold).float()

        for idx in range(len(token_predictions)):
            print(batch[-1]['Entry'][idx])
            print(torch.nonzero(token_predictions[idx]))

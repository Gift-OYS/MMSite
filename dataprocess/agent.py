from prot2text_model.Model import Prot2TextModel
from prot2text_model.tokenization_prot2text import Prot2TextTokenizer
import torch
import pandas as pd
from tqdm import tqdm
import json

model_path = '/path/to/pretrained_weights/esm2text_base'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = Prot2TextModel.from_pretrained(model_path)
tokenizer = Prot2TextTokenizer.from_pretrained(model_path)

def prot2text(seq):
    encoded = model.generate_protein_description(protein_sequence=seq, device=device)[0]
    encoded = encoded[encoded != 50256]
    encoded = encoded[encoded != 50257]
    encoded = encoded[encoded != 50258]
    description = tokenizer.decode(encoded)
    return description

train = pd.read_csv('/path/to/datasets/train.tsv', sep='\t')
valid = pd.read_csv('/path/to/datasets/valid.tsv', sep='\t')
test = pd.read_csv('/path/to/datasets/test.tsv', sep='\t')
total = pd.concat([train, valid, test], ignore_index=True)
total.reset_index(drop=True, inplace=True)
entry_prot2text = {}

for idx in tqdm(range(len(total))):
    seq = total.loc[idx, "Sequence"]
    text = prot2text(seq)
    entry_prot2text[total.loc[idx, "Entry"]] = text
    
with open('/path/to/datasets/generated_desc.json', 'w') as f:
    json.dump(entry_prot2text, f)

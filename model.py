import torch
import torch.nn as nn
from transformers import AutoModel

from utils.util_functions import cos_sim, kl_loss

class AP_align_fuse(torch.nn.Module):
        
    def __init__(self, config, hidden_size=256):
        super(AP_align_fuse, self).__init__()
        self.config = config
        self.tau = config.tau
        self.text_model = AutoModel.from_pretrained(f'{config.model.model_dir}/{config.model.pubmed_version}')
        self.seq_model = AutoModel.from_pretrained(f'{config.model.model_dir}/{config.model.esm_version}')

        embedding_dim = 1280
        self.num_attr = 17
        self.function_len = 128

        if hasattr(self, 'seq_model'):
            for param in self.seq_model.parameters():
                param.requires_grad = False
        for param in self.text_model.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(0.1)
        self.num_labels = 2

        self.project = nn.Linear(embedding_dim, 768)

        self.texts_encoder = nn.ModuleList([nn.TransformerEncoderLayer(d_model=768, nhead=4, dropout=0.1) for _ in range(self.num_attr)])
        self.text_suffix_encoder = nn.TransformerEncoderLayer(d_model=768, nhead=4, dropout=0.1)
        self.text_suffix_transformer = nn.TransformerEncoder(self.text_suffix_encoder, num_layers=2)
        self.text_crosses = nn.ModuleList([nn.MultiheadAttention(768, num_heads=4, dropout=0.1, batch_first=True) for _ in range(4)])
        self.norms = nn.ModuleList([nn.LayerNorm(768) for _ in range(4)])

        self.seq_suffix_encoder = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dropout=0.1)
        self.seq_suffix_transformer = nn.TransformerEncoder(self.seq_suffix_encoder, num_layers=2)

        self.share_encoder = nn.TransformerEncoderLayer(d_model=768, nhead=8, dropout=0.1)
        self.share_transformer = nn.TransformerEncoder(self.share_encoder, num_layers=4)

        self.token_suffix_encoder_res = nn.TransformerEncoderLayer(d_model=768+embedding_dim, nhead=4, dropout=0.1)
        self.token_suffix_transformer_res = nn.TransformerEncoder(self.token_suffix_encoder_res, num_layers=2)
        
        self.token_suffix_encoder = nn.TransformerEncoderLayer(d_model=768, nhead=4, dropout=0.1)
        self.token_suffix_transformer = nn.TransformerEncoder(self.token_suffix_encoder, num_layers=2)

        self.fusion_cross = nn.MultiheadAttention(768, num_heads=4, dropout=0.1, batch_first=True)
        
        self.bn2_res = nn.BatchNorm1d(hidden_size)
        self.fc2_res = nn.Linear(768+embedding_dim, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(768, hidden_size)
        self.classifier_token = nn.Linear(hidden_size, 2)

    def forward(self, anchor_text_input_ids=None,
                anchor_text_attention_mask=None,
                anchor_seq_input_ids=None,
                anchor_seq_attention_mask=None,
                test=False):

        if not test:
            anchor_text_outputs = []
            for i in range(len(anchor_text_input_ids)):
                input_id = anchor_text_input_ids[i]
                attention_mask = anchor_text_attention_mask[i]
                text_output = self._get_model_output(input_id, attention_mask, 'text')
                anchor_text_outputs.append(text_output)
        else:
            anchor_text_branch_output_o = self._get_model_output(anchor_text_input_ids, anchor_text_attention_mask, 'text')

        anchor_seq_outputs = self._get_model_output(anchor_seq_input_ids, anchor_seq_attention_mask, 'seq')
        anchor_seq_outputs = anchor_seq_outputs[0]

        if not test:
            anchor_text_branch_output_o = self._get_text_branch_output(anchor_text_outputs)
            anchor_text_branch_output = anchor_text_branch_output_o
        else:
            anchor_text_branch_output = anchor_text_branch_output_o[0]

        anchor_seq_branch_output = self.project(anchor_seq_outputs)
        
        anchor_seq_branch_output = self.share_transformer(anchor_seq_branch_output)
        anchor_text_branch_output = self.share_transformer(anchor_text_branch_output)

        cross_modal_losses = self._softlabel_loss_3d(anchor_seq_branch_output, anchor_text_branch_output, tau=self.tau)
        fusion = self.fusion_cross(anchor_seq_branch_output, anchor_text_branch_output, anchor_text_branch_output)
        
        anchor_seq_outputs = self.seq_suffix_transformer(anchor_seq_outputs)
        
        fusion = torch.cat([anchor_seq_outputs, fusion[0]], dim=2)
        seq_pred = self.token_suffix_transformer_res(fusion)
        seq_pred = torch.relu(self.bn2_res(self.fc2_res(seq_pred).permute(0, 2, 1)).permute(0, 2, 1))
        seq_pred = self.classifier_token(seq_pred)[:,:,self.num_labels-1].squeeze(-1)

        return {
            'token_logits': torch.sigmoid(seq_pred),
            'contrastive_loss': cross_modal_losses,
        }

    def _softlabel_loss_3d(self, seq_features, text_features, tau):
        seq_features = seq_features.mean(dim=1)
        text_features = text_features.mean(dim=1)
        seq_sim, text_sim = cos_sim(seq_features, seq_features), cos_sim(text_features, text_features)
        logits_per_seq, logits_per_text = self._get_similarity(seq_features, text_features)
        cross_modal_loss = (kl_loss(logits_per_seq, seq_sim, tau=tau) + 
                            kl_loss(logits_per_text, text_sim, tau=tau)) / 2.0
        return cross_modal_loss

    def _get_similarity(self, image_features, text_features):
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logits_per_image = image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text

    def _get_model_output(self, input_ids, attention_mask, modal):
        if modal == 'text':
            model = self.text_model
        else:
            model = self.seq_model
        return model(input_ids=input_ids, attention_mask=attention_mask)
    
    def _get_text_branch_output(self, text_outputs):
        texts_output = [self.texts_encoder[i](text_outputs[i][0]) for i in range(self.num_attr)]
        texts_output_cls = [texts_output[idx][:, 0, :].unsqueeze(1) for idx in range(len(texts_output)) if idx != 3]
        texts_output_cls = torch.cat(texts_output_cls, dim=1)
        texts_output_cls = self.text_suffix_transformer(texts_output_cls)
        text_func = texts_output[3]
        x = texts_output_cls
        for i in range(4):
            _x = x
            x = self.text_crosses[i](x, text_func, text_func)
            x = self.norms[i](x[0] + _x)
        text_branch_output = x
        return text_branch_output

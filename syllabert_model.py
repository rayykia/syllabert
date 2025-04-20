import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from transformers import HubertModel
from findsylls import segment_waveform

class SyllaBERTEncoder(nn.Module):
    def __init__(self,
                 #input_dim=1,
                 #embed_dim=768,
                 #num_heads=12,
                 #dropout=0.1,
                 hubert_pretrained_model: str = None):
        super().__init__()
        # Load HuBERT pretrained model
        hubert_name = hubert_pretrained_model or "facebook/hubert-base-ls960"
        self.hubert = HubertModel.from_pretrained(hubert_name)
        # conv feature extractor from HuBERT
        self.feature_extractor = self.hubert.feature_extractor
        self.conv_stride = 320  # total stride of conv layers
        self.feature_projection = self.hubert.feature_projection
        # reuse pretrained HuBERT transformer encoder (all layers)
        self.transformer = self.hubert.encoder
        self.layer_norm = self.transformer.layer_norm if hasattr(self.transformer, 'layer_norm') else nn.LayerNorm(self.hubert.config.hidden_size)
        self.projection = nn.Sequential(nn.Linear(768, 100), nn.Softmax())

    def forward(self, waveforms: torch.Tensor, sampling_rate: int = 16000):
        B, T = waveforms.size()
        device = waveforms.device
        # conv features
        feats = self.feature_extractor(waveforms)  # (B, C, T')
        feats = feats.transpose(1, 2)              # (B, T', C)
        T_frames = feats.size(1)
        pooled_list = []

        num_syll = 0
        for b in range(B):
            wav_np = waveforms[b].detach().cpu().numpy()
            print(f'{wav_np.shape = }')
            sylls, _, _ = segment_waveform(wav_np, sampling_rate)
            num_syll += len(sylls)

            segments = []
            for s_sec, _, e_sec in sylls:
                s_fr = int(s_sec * sampling_rate / self.conv_stride)
                e_fr = int(e_sec * sampling_rate / self.conv_stride)
                segments.append((max(0, s_fr), min(T_frames, max(s_fr+1, e_fr))))
            if not segments:
                reps = feats[b].mean(dim=0, keepdim=True)
            else:
                reps = torch.stack([feats[b, s:e].mean(dim=0) for s,e in segments])
            pooled_list.append(reps)

        print(f'{num_syll =}')
        # pad pools and mask
        max_syll = max(r.size(0) for r in pooled_list)
        C = feats.size(2)
        padded = torch.zeros(B, max_syll, C, device=device)
        mask = torch.ones(B, max_syll, dtype=torch.bool, device=device)
        for b, reps in enumerate(pooled_list):
            L = reps.size(0)

            padded[b,:L] = reps
            mask[b,:L] = False
        # positions
        # positions = torch.arange(max_syll, device=device).unsqueeze(0)
        # padded = padded + self.pos_embedding(positions)
        # transformer
        x = padded
        x = self.feature_projection(x)
        x = self.transformer(x, attention_mask=mask).last_hidden_state
        # x = x.transpose(0,1)
        x = self.layer_norm(x)
        # project and split
        outputs = []
        total_l = 0
        for b in range(B):
            L = (~mask[b]).sum().item()
            total_l += L
            print(f'{L = }')
            # print(f'{x[b].shape = }')
            outputs.append(self.projection(x[b,:L]))

        print(f'{total_l =}')
        return outputs
    
# alias for compatibility
SyllaBERT = SyllaBERTEncoder


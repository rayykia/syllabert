# syllabert_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class SyllaBERTEncoder(nn.Module):
    def __init__(self, input_dim=1, embed_dim=768, num_layers=12, num_heads=12, dropout=0.1, conv_layers=None):
        super().__init__()
        if conv_layers is None:
            conv_layers = [
                (512, 10, 5),
                (512, 3, 2),
                (512, 3, 2),
                (512, 3, 2),
                (512, 2, 2),
                (512, 2, 2),
                (embed_dim, 2, 2)
            ]
        self.conv_layers = conv_layers
        self.feature_extractor = nn.Sequential(
            *[nn.Sequential(
                nn.Conv1d(in_channels=input_dim if i == 0 else conv_layers[i-1][0],
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride),
                nn.ReLU())
              for i, (out_channels, kernel_size, stride) in enumerate(conv_layers)]
        )

        self.pos_embedding = nn.Embedding(5000, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, src_key_padding_mask=None):
        original_mask = src_key_padding_mask
        # x is already shaped (B, 1, T)
        for layer in self.feature_extractor:
            if src_key_padding_mask is not None and isinstance(layer, nn.Sequential):
                stride = layer[0].stride[0]
                src_key_padding_mask = src_key_padding_mask[:, ::stride]
        x = self.feature_extractor(x)  # (B, embed_dim, T')
        x = x.transpose(1, 2)  # (B, T', embed_dim)

        positions = torch.arange(0, x.size(1), dtype=torch.long, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)

        x = x.transpose(0, 1)  # (T', B, embed_dim)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = x.transpose(0, 1)  # (B, T', embed_dim)
        x = self.layer_norm(x)
        return x

class SyllaBERT(nn.Module):
    def __init__(self, input_dim=1, embed_dim=768, num_layers=12, num_heads=12, num_classes=100):
        super().__init__()
        self.encoder = SyllaBERTEncoder(input_dim, embed_dim, num_layers, num_heads)
        self.projection = nn.Linear(embed_dim, num_classes)

    def forward(self, x, syllable_segments):
        encoded = self.encoder(x)
        pooled = []
        for b, segments in enumerate(syllable_segments):
            reps = [encoded[b, start:end].mean(dim=0) for start, end in segments]
            pooled.append(torch.stack(reps))
        pooled = torch.cat(pooled, dim=0)  # ensure pooled matches target shape
        logits = self.projection(pooled)
        return logits

    def compute_loss(self, logits, target):
        # logits: (total_syllables, num_classes)
        # target: (total_syllables,)
        return F.cross_entropy(logits, target, ignore_index=-100)

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path, map_location=None):
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)
        print(f"Model loaded from {path}")

    def required_input_length(self):
        length = 1
        for layer in reversed(self.encoder.feature_extractor):
            if isinstance(layer, nn.Sequential):
                conv = layer[0]  # assumes Conv1d followed by ReLU
                kernel_size = conv.kernel_size[0]
                stride = conv.stride[0]
                length = (length - 1) * stride + kernel_size
        return length

    def get_conv_layers(self):
        return self.encoder.conv_layers


def hubert_style_mask(batch_size, seq_len, mask_prob=0.065, mask_length=10):
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    num_mask = int(mask_prob * seq_len / mask_length + 0.5)
    for b in range(batch_size):
        mask_starts = torch.randperm(seq_len - mask_length)[:num_mask]
        for start in mask_starts:
            mask[b, start:start + mask_length] = True
    return mask

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from findsylls import segment_waveform

class AttentionPool(nn.Module):
    """
    Attention-based pooling over variable-length frame segments.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.attn_vector = nn.Parameter(torch.randn(embed_dim))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [N, C]
        if frames.size(0) == 0:
            return torch.zeros(frames.size(1), device=frames.device)
        scores = frames.matmul(self.attn_vector)
        weights = torch.softmax(scores, dim=0)
        return (weights.unsqueeze(-1) * frames).sum(dim=0)

class SyllaBERT(nn.Module):
    """
    SyllaBERT auto-segments audio into syllables using segment_waveform,
    attention-pools frame features into syllable tokens, applies optional masking,
    and predicts cluster labels per syllable.

    Input to forward():
      x: waveform tensor [B,1,T]
      apply_mask: mask some syllable tokens
      mask_prob: fraction of tokens to mask

    Output:
      logits: [B, S_max, num_classes]
      mask_tokens: [B, S_max]
      pad_mask: [B, S_max]
    """
    def __init__(self,
                 input_dim=1,
                 embed_dim=768,
                 num_layers=12,
                 num_heads=12,
                 num_classes=100,
                 dropout=0.1,
                 max_syllables=200,
                 sampling_rate=16000):
        super().__init__()
        # convolutional encoder
        conv_spec = [
            (512, 10, 5), (512, 3, 2), (512, 3, 2),
            (512, 3, 2), (512, 2, 2), (512, 2, 2), (embed_dim, 2, 2)
        ]
        layers = []
        self.stride = 1
        for i, (out_c, k, s) in enumerate(conv_spec):
            in_c = input_dim if i == 0 else conv_spec[i-1][0]
            layers += [nn.Conv1d(in_c, out_c, k, stride=s), nn.ReLU()]
            self.stride *= s
        self.feature_extractor = nn.Sequential(*layers)
        # attention pooling
        self.attn_pool = AttentionPool(embed_dim)
        # mask token embedding
        self.mask_embedding = nn.Parameter(torch.randn(embed_dim))
        # positional embeddings for syllable tokens
        self.pos_embedding = nn.Embedding(max_syllables, embed_dim)
        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)
        # projection to pseudo-label classes
        self.projection = nn.Linear(embed_dim, num_classes)
        self.sr = sampling_rate

    def forward(self,
                x: torch.Tensor,
                apply_mask: bool = False,
                mask_prob: float = 0.65) -> tuple:
        B = x.size(0)
        # extract conv features
        f = self.feature_extractor(x)      # [B, C, T']
        feats = f.transpose(1, 2)          # [B, T', C]
        Tprime, C = feats.size(1), feats.size(2)
        # segment audio into syllables
        all_segs = []
        for b in range(B):
            waveform = x[b,0].detach().cpu().numpy()
            sylls, _, _ = segment_waveform(waveform, self.sr)
            segs = []
            for (t0, _, t1) in sylls:
                s = int(t0 * self.sr) // self.stride
                e = (int(t1 * self.sr) + self.stride - 1) // self.stride
                s = max(0, min(s, Tprime))
                e = max(s+1, min(e, Tprime))
                segs.append((s, e))
            all_segs.append(segs)
        # determine max syllables
        Smax = max(len(segs) for segs in all_segs)
        # pool to tokens
        token_list, pad_masks = [], []
        for b, segs in enumerate(all_segs):
            reps = []
            for (s, e) in segs:
                reps.append(self.attn_pool(feats[b, s:e]))
            pad_len = Smax - len(segs)
            if pad_len > 0:
                reps += [torch.zeros(C, device=x.device)] * pad_len
                pad_mask = torch.tensor([False]*len(segs) + [True]*pad_len, device=x.device)
            else:
                pad_mask = torch.zeros(Smax, dtype=torch.bool, device=x.device)
            token_list.append(torch.stack(reps))
            pad_masks.append(pad_mask)
        tokens = torch.stack(token_list, dim=0)    # [B, Smax, C]
        pad_mask = torch.stack(pad_masks, dim=0)   # [B, Smax]
        # optional masking
        mask_tokens = None
        if apply_mask:
            mask_tokens = torch.zeros((B, Smax), dtype=torch.bool, device=x.device)
            for b, segs in enumerate(all_segs):
                n = len(segs)
                m = int(n*mask_prob + 0.5)
                if m > 0:
                    idxs = torch.randperm(n, device=x.device)[:m]
                    mask_tokens[b, idxs] = True
            m_e = self.mask_embedding.unsqueeze(0).unsqueeze(0).expand_as(tokens)
            tokens = torch.where(mask_tokens.unsqueeze(-1), m_e, tokens)
        # add positional
        pos = torch.arange(Smax, device=x.device).unsqueeze(0)
        tokens = tokens + self.pos_embedding(pos)
        # transformer
        y = tokens.transpose(0,1)              # [Smax, B, C]
        y = self.transformer(y, src_key_padding_mask=pad_mask)
        y = y.transpose(0,1)                   # [B, Smax, C]
        y = self.layer_norm(y)
        logits = self.projection(y)            # [B, Smax, num_classes]
        return logits, mask_tokens, pad_mask

    def compute_loss(self,
                     logits: torch.Tensor,
                     targets: torch.LongTensor,
                     mask_tokens: torch.BoolTensor) -> torch.Tensor:
        ml = logits[mask_tokens]
        mt = targets[mask_tokens]
        if ml.numel() == 0:
            return torch.tensor(0., device=ml.device)
        return F.cross_entropy(ml, mt, ignore_index=-100)

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: str, map_location=None):
        st = torch.load(path, map_location=map_location)
        self.load_state_dict(st)
        print(f"Loaded checkpoint from {path}")

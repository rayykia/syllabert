import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class SyllaBERTEncoder(nn.Module):
    """
    Convolutional encoder: raw waveform -> frame-level features
    """
    def __init__(self,
                 input_dim=1,
                 embed_dim=768,
                 conv_layers=None):
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
        layers = []
        for i, (out_c, k, s) in enumerate(conv_layers):
            in_c = input_dim if i == 0 else conv_layers[i-1][0]
            layers += [nn.Conv1d(in_c, out_c, k, stride=s), nn.ReLU()]
        self.feature_extractor = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B,1,T] -> [B,C,T'] -> [B,T',C]
        f = self.feature_extractor(x)
        return f.transpose(1, 2)


def hubert_style_mask_segments(boundary_targets, mask_prob):
    """
    Generate a boolean mask per frame by randomly masking whole syllable segments.
    boundary_targets: [B, T'] binary tensor indicating boundaries after frame t
    Returns: mask_indices [B, T']
    """
    B, T = boundary_targets.shape
    mask = torch.zeros((B, T), dtype=torch.bool, device=boundary_targets.device)
    for b in range(B):
        # find segment boundaries
        # boundaries at True, so segment spans between them
        boundaries = boundary_targets[b].nonzero(as_tuple=False).squeeze(1).tolist()
        # ensure start=0, end=T
        seg_points = [0] + boundaries + [T]
        # build list of (start,end)
        segments = [(seg_points[i], seg_points[i+1]) for i in range(len(seg_points)-1)]
        # sample segments to mask
        num_mask = int(len(segments) * mask_prob + 0.5)
        if num_mask > 0:
            idxs = torch.randperm(len(segments), device=boundary_targets.device)[:num_mask]
            for i in idxs:
                s,e = segments[i]
                mask[b, s:e] = True
    return mask

class SyllaBERT(nn.Module):
    """
    SyllaBERT with syllable-level masking based on boundary targets.
    """
    def __init__(self,
                 input_dim=1,
                 embed_dim=768,
                 num_layers=12,
                 num_heads=12,
                 num_classes=100,
                 dropout=0.1,
                 max_frames=5000):
        super().__init__()
        # encoder
        self.encoder = SyllaBERTEncoder(input_dim, embed_dim)
        # mask embedding
        self.mask_embedding = nn.Parameter(torch.randn(embed_dim))
        # positional embeddings
        self.pos_embedding = nn.Embedding(max_frames, embed_dim)
        # transformer
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                               nhead=num_heads,
                                               dropout=dropout)
        self.transformer = nn.TransformerEncoder(enc_layer,
                                                 num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)
        # projection head for cluster prediction
        self.projection = nn.Linear(embed_dim, num_classes)
        # boundary head for segmentation
        self.boundary_pred = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(embed_dim, 1, 1),
            nn.Sigmoid()
        )

    def forward(self,
                x,
                boundary_targets=None,
                apply_mask=False,
                mask_prob=0.65):
        """
        Args:
          x: [B,1,T] raw waveform
          boundary_targets: [B,T'] binary frame-level boundaries (gold) or None
          apply_mask: mask whole syllable segments
          mask_prob: fraction of syllables to mask
        Returns:
          logits: [B,T',num_classes]
          mask_indices: [B,T'] boolean
          boundary_prob: [B,T']
        """
        # extract frame features
        feats = self.encoder(x)  # [B,T',C]
        B, T, C = feats.shape
        # boundary prediction
        bp = feats.transpose(1,2)  # [B,C,T']
        boundary_prob = self.boundary_pred(bp).squeeze(1)  # [B,T']
        # determine mask indices
        mask_indices = None
        if apply_mask:
            if boundary_targets is None:
                raise ValueError("boundary_targets required for syllable-level masking")
            mask_indices = hubert_style_mask_segments(boundary_targets, mask_prob)
            # mask embedding
            mask_e = self.mask_embedding.unsqueeze(0).unsqueeze(0).expand(B, T, C)
            feats = torch.where(mask_indices.unsqueeze(-1), mask_e, feats)
        # positional embedding
        pos = torch.arange(T, device=feats.device).unsqueeze(0)
        feats = feats + self.pos_embedding(pos)
        # transformer
        y = feats.transpose(0,1)  # [T',B,C]
        y = self.transformer(y)
        y = y.transpose(0,1)  # [B,T',C]
        y = self.layer_norm(y)
        # cluster logits
        logits = self.projection(y)  # [B,T',num_classes]
        return logits, mask_indices, boundary_prob

    def compute_cluster_loss(self, logits, targets, mask_indices):
        """
        CE loss on masked syllable frames only; targets: [B,T']
        """
        if mask_indices is None:
            raise ValueError("mask_indices required for cluster loss")
        ml = logits[mask_indices]
        mt = targets[mask_indices]
        if ml.numel()==0:
            return torch.tensor(0., device=ml.device)
        return F.cross_entropy(ml, mt, ignore_index=-100)

    def compute_boundary_loss(self, boundary_prob, boundary_targets):
        """
        BCE loss for boundary prediction
        """
        return F.binary_cross_entropy(boundary_prob, boundary_targets.float())

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path, map_location=None):
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state)
        print(f"Loaded checkpoint from {path}")

    def required_input_length(self):
        # implement if needed
        raise NotImplementedError

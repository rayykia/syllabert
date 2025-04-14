import torch
import torch.nn as nn
import torch.nn.functional as F
import os

def hubert_style_mask(batch_size, seq_len, mask_prob, mask_length):
    """
    Fallback masking function that masks random contiguous spans if syllable segments are not provided.
    Returns a boolean mask of shape (batch_size, seq_len).
    """
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    num_mask = int(mask_prob * seq_len / mask_length + 0.5)
    for b in range(batch_size):
        if seq_len - mask_length > 0:
            mask_starts = torch.randperm(seq_len - mask_length)[:num_mask]
            for start in mask_starts:
                mask[b, start:start + mask_length] = True
    return mask


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
        
        # Learnable mask embedding to replace masked features.
        self.mask_embedding = nn.Parameter(torch.randn(embed_dim))

    def forward(self, x, src_key_padding_mask=None, apply_mask=False, mask_prob=0.65, mask_length=10, syllable_segments=None):
        """
        Args:
            x (torch.Tensor): Input waveform tensor of shape (B, 1, T).
            src_key_padding_mask (torch.BoolTensor): Optional padding mask.
            apply_mask (bool): Whether to apply masking.
            mask_prob (float): Probability of masking. For syllable-level masking, this is applied per syllable.
            mask_length (int): When syllable segments are not provided, use this span length.
            syllable_segments (list or None): For each batch item, a list of tuples (start, end) indicating 
                                              syllable boundaries on the convolutional output timeline.
        Returns:
            x (torch.Tensor): Encoded features of shape (B, T', embed_dim).
        """
        # Adjust padding mask across convolutional layers.
        for layer in self.feature_extractor:
            if src_key_padding_mask is not None and isinstance(layer, nn.Sequential):
                stride = layer[0].stride[0]
                src_key_padding_mask = src_key_padding_mask[:, ::stride]
        
        # Extract features using convolutional layers.
        x = self.feature_extractor(x)  # Shape: (B, embed_dim, T')
        x = x.transpose(1, 2)          # Shape: (B, T', embed_dim)

        # --- Apply masking on the convolutional output BEFORE positional encoding and transformer ---
        if apply_mask:
            B, T, C = x.shape
            if syllable_segments is not None:
                # Build a boolean mask tensor over convolution outputs for syllable-level masking.
                segment_mask = torch.zeros((B, T), dtype=torch.bool, device=x.device)
                for b, segments in enumerate(syllable_segments):
                    for (start, end) in segments:
                        # Ensure segment indices are within bounds.
                        start = max(0, start)
                        end = min(T, end)
                        if start < end and torch.rand(1).item() < mask_prob:
                            segment_mask[b, start:end] = True
                # Expand mask to match the feature dimensions.
                mask_embed = self.mask_embedding.unsqueeze(0).unsqueeze(0).expand_as(x)
                # Use torch.where to perform out-of-place replacement.
                x = torch.where(segment_mask.unsqueeze(-1), mask_embed, x)
            else:
                # Fallback to frame-level contiguous masking.
                mask = hubert_style_mask(B, T, mask_prob, mask_length).to(x.device)
                mask_embed = self.mask_embedding.unsqueeze(0).unsqueeze(0).expand(B, T, C)
                x = torch.where(mask.unsqueeze(-1), mask_embed, x)

        # Add positional embeddings.
        positions = torch.arange(0, x.size(1), dtype=torch.long, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)

        # Pass through the transformer encoder.
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

    def forward(self, x, syllable_segments, apply_mask=False, mask_prob=0.65, mask_length=10):
        """
        Args:
            x (torch.Tensor): Input waveform tensor of shape (B, 1, T).
            syllable_segments (list): For each batch item, a list of (start, end) tuples for syllable boundaries.
            apply_mask (bool): Whether to apply syllable-level masking.
            mask_prob (float): Probability of masking each syllable.
            mask_length (int): Ignored if syllable segments are provided.
        Returns:
            logits (torch.Tensor): Prediction logits (one per syllable), e.g., for cluster classification.
        """
        # Forward pass through the encoder with masking applied before the transformer.
        encoded = self.encoder(
            x,
            apply_mask=apply_mask,
            mask_prob=mask_prob,
            mask_length=mask_length,
            syllable_segments=syllable_segments
        )
        # Pool the encoder outputs over each syllable region.
        pooled = []
        for b, segments in enumerate(syllable_segments):
            reps = [encoded[b, start:end].mean(dim=0) for start, end in segments]
            pooled.append(torch.stack(reps))
        pooled = torch.cat(pooled, dim=0)  # Shape: (total_syllables, embed_dim)
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
                conv = layer[0]  # Assumes Conv1d followed by ReLU.
                kernel_size = conv.kernel_size[0]
                stride = conv.stride[0]
                length = (length - 1) * stride + kernel_size
        return length

    def get_conv_layers(self):
        return self.encoder.conv_layers

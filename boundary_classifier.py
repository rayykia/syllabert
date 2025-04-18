import torch
import torch.nn as nn
import numpy as np
from transformers import HubertModel
from findsylls import segment_waveform

class HubertSyllableBoundary(nn.Module):
    """
    Self‐supervised HuBERT syllable boundary classifier.
    During training, generates pseudo‐labels via `segment_waveform` and
    computes a boundary classification loss. In evaluation (inference) mode,
    it returns only the boundary logits.
    """
    def __init__(self,
                 pretrained_model_name: str = "facebook/hubert-base-ls960",
                 conv_stride: int = 5 * (2 ** 6)):
        super().__init__()
        # Load pretrained HuBERT (includes conv front‐end)
        self.encoder = HubertModel.from_pretrained(pretrained_model_name)
        hidden_size = self.encoder.config.hidden_size
        # Binary head: non‐boundary vs boundary
        self.boundary_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2)
        )
        # stride between raw samples and frame outputs
        self.conv_stride = conv_stride
        # loss function for training
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self,
                waveforms: torch.Tensor,
                sampling_rate: int = 16000):
        """
        Args:
            waveforms: (B, T_samples) raw audio batch
            sampling_rate: sampling rate of audio (Hz)

        Returns:
            If training mode:
                loss (torch.Tensor): cross‐entropy loss against pseudo‐labels
                logits (torch.Tensor): (B, T_frames, 2) boundary logits
            If eval mode:
                logits (torch.Tensor): (B, T_frames, 2) boundary logits
        """
        device = waveforms.device
        # 1) encode raw audio to frame representations
        outputs = self.encoder(waveforms, attention_mask=None)
        hidden = outputs.last_hidden_state  # (B, T_frames, hidden)
        # 2) predict logits for boundary vs non-boundary
        logits = self.boundary_head(hidden)  # (B, T_frames, 2)

        # In inference mode, skip pseudo-labeling and loss
        if not self.training:
            return logits

        # In training mode, generate pseudo-labels via findsylls
        B, T_frames, _ = logits.shape
        labels = []
        # need numpy waveforms for segment_waveform
        wf_np = waveforms.detach().cpu().numpy()
        for b in range(B):
            sylls, _, _ = segment_waveform(wf_np[b], sampling_rate)
            label = torch.zeros(T_frames, dtype=torch.long)
            for (s_sec, _, _) in sylls:
                start_frame = int(s_sec * sampling_rate / self.conv_stride)
                if 0 <= start_frame < T_frames:
                    label[start_frame] = 1
            labels.append(label)
        labels = torch.stack(labels, dim=0).to(device)

        # 3) compute frame-level cross-entropy loss
        loss = self.loss_fn(
            logits.view(-1, 2),
            labels.view(-1)
        )
        return loss, logits

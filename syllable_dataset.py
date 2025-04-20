# syllable_dataset.py

import torch
import json
import librosa
from collections import defaultdict
from findsylls import segment_waveform

class SyllableDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path, samplerate=16000):
        self.samplerate = samplerate
        with open(manifest_path, 'r') as f:
            entries = [json.loads(line) for line in f]
        # group by utterance
        self.utterances = defaultdict(list)
        for e in entries:
            self.utterances[e["utterance_id"]].append(e)
        # sort by segment_index
        for utt in self.utterances:
            self.utterances[utt].sort(key=lambda x: x["segment_index"])
        self.keys = list(self.utterances.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        utt_id = self.keys[idx]
        segs = self.utterances[utt_id]
        # load full waveform for that utterance
        path = segs[0]["audio_file"]
        y, _ = librosa.load(path, sr=self.samplerate)
        # crop to the span of all segments
        start = int(min(e["segment_start"] for e in segs) * self.samplerate)
        end   = int(max(e["segment_end"]   for e in segs) * self.samplerate)
        wav = y[start:end]
        wav = torch.from_numpy(wav).float().unsqueeze(0)  # [1, T]
        # collect cluster IDs
        labels = [e.get("cluster_id", -100) for e in segs]
        labels = torch.tensor(labels, dtype=torch.long)   # [S]
        return wav, labels

def collate_syllable_utterances(batch):
    """
    Pads waveforms to max T, and label sequences to max S.
    Returns:
      waves:      [B,1,T_max]
      labels:     [B,S_max] filled with -100
      pad_mask:   [B,S_max] True where padded (ignore positions)
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, None

    waves, labs = zip(*batch)

    num_syll_ds = 0
    for wav in waves:
        wav = wav.detach().cpu().numpy().flatten()
        print(f'{wav.shape =}')
        syll, _, _ = segment_waveform(wav)
        num_syll_ds += len(syll)

    print(f'{num_syll_ds = }')
    
    print(f'{torch.cat(labs).shape =}')

    B = len(waves)
    # pad waveforms
    T_max = max(w.size(1) for w in waves)
    padded_w = torch.zeros(B, T_max)
    for i, w in enumerate(waves):
        padded_w[i, :w.size(1)] = w

    # pad labels
    S_max = max(l.numel() for l in labs)
    padded_l = torch.full((B, S_max), -100, dtype=torch.long)
    pad_mask = torch.ones((B, S_max), dtype=torch.bool)
    total_l = 0
    for i, l in enumerate(labs):
        L = l.size(0)
        padded_l[i, :L] = l
        pad_mask[i, :L] = False
        total_l += L
    print(f'{total_l = }')
    # print(f'{torch.nn.functional.one_hot(torch.cat(labs), num_classes=100).shape =}')
    one_hot_label = torch.nn.functional.one_hot(torch.cat(labs), num_classes=100)
    return padded_w, padded_l, pad_mask, one_hot_label

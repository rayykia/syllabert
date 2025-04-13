# syllable_dataset.py

import torch
import json
import librosa
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import math

class SyllableDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_path, samplerate=16000):
        self.samplerate = samplerate

        with open(manifest_path, 'r') as f:
            entries = [json.loads(line) for line in f]

        self.utterances = defaultdict(list)
        for entry in entries:
            utt_id = entry.get("utterance_id")
            if utt_id is not None:
                self.utterances[utt_id].append(entry)

        for utt in self.utterances:
            self.utterances[utt].sort(key=lambda e: e["segment_index"])

        self.utterance_keys = list(self.utterances.keys())

    def __len__(self):
        return len(self.utterance_keys)

    def __getitem__(self, idx):
        utt_id = self.utterance_keys[idx]
        segments = self.utterances[utt_id]

        audio_file = segments[0]["audio_file"]
        audio, _ = librosa.load(audio_file, sr=self.samplerate)

        full_start = int(min(entry["segment_start"] for entry in segments) * self.samplerate)
        full_end = int(max(entry["segment_end"] for entry in segments) * self.samplerate)
        waveform = torch.tensor(audio[full_start:full_end], dtype=torch.float32).view(1, -1)

        labels = [entry.get("cluster_id", -100) for entry in segments]
        targets = torch.tensor(labels, dtype=torch.long)

        # Convert syllable start/end times into frame indices after conv stack (stride=320)
        conv_stride = 320
        segments_frames = []
        for entry in segments:
            s = int((entry["segment_start"] * self.samplerate - full_start) / conv_stride)
            e = int((entry["segment_end"] * self.samplerate - full_start) / conv_stride)
            e = max(s + 1, e)  # ensure at least 1 frame
            segments_frames.append((s, e))

        return waveform, targets, segments_frames

def collate_syllable_utterances(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None, None, None

    waveforms, targets, segments = zip(*batch)
    lengths = [x.size(1) for x in waveforms]
    max_len = max(lengths)

    padded_waveforms = torch.zeros((len(waveforms), 1, max_len), dtype=torch.float32)
    for i, x in enumerate(waveforms):
        padded_waveforms[i, :, :x.size(1)] = x

    all_targets = [t for t in targets]
    return padded_waveforms, all_targets, segments

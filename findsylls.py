#!/usr/bin/env python
"""
findsylls.py

This module implements syllable segmentation using an amplitude-spectrum based method 
and provides routines to evaluate segmentation against aligned TextGrid syllable annotations.

You can now call `segment_waveform(audio, sr, **kwargs)` directly on a NumPy waveform,
or `segment_audio(path, sr, **kwargs)` to load from disk.

Dependencies:
    - numpy
    - matplotlib (for plotting)
    - scipy
    - librosa
    - praatio (for TextGrid parsing)

Usage:
    from findsylls import segment_waveform, segment_audio, evaluate_segmentation
    syllables, t, A = segment_waveform(wav_array, sr=16000)
    syllables2, t2, A2 = segment_audio("audio.wav")

    # For evaluation:
    precision, recall, f1 = evaluate_segmentation(audio_list, tg_list)
"""
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, filtfilt
from scipy.signal.windows import hamming
import librosa
from praatio import textgrid  # pip install praatio


def segment_waveform(audio: np.ndarray,
                     sr: int = 16000,
                     nfft: int = 256,
                     window_length: int = 256,
                     step: int = 160,
                     min_peak: float = 0.04,
                     smoothing_window_samples: int = 7,
                     pivot_freq: float = 3000,
                     show_plots: bool = False) -> tuple:
    """
    Segment a raw waveform into syllables using amplitude-spectrum method.

    Returns:
        syllables: List of (start_time, peak_time, end_time) in seconds.
        t: time vector for envelope
        A: normalized amplitude envelope
    """
    # Compute spectrogram magnitude
    noverlap = window_length - step
    f, t, Sxx = spectrogram(audio,
                             fs=sr,
                             nfft=nfft,
                             window='hamming',
                             nperseg=window_length,
                             noverlap=noverlap,
                             mode='magnitude')
    t = t + (nfft / sr) / 2.0

    # Determine pivot bin
    freq_res = sr / nfft
    pivot_bin = int(np.ceil(pivot_freq / freq_res))

    # Low vs. high energy difference
    low_energy = np.sum(Sxx[1:pivot_bin, :], axis=0)
    high_energy = np.sum(Sxx[pivot_bin:(nfft//2), :], axis=0)
    diff_signal = low_energy - high_energy
    diff_signal[diff_signal < 0] = 0

    # Smooth via Hamming window
    win = hamming(smoothing_window_samples)
    smooth_signal = filtfilt(win, [1.0], diff_signal)

    # Normalize envelope
    A = smooth_signal
    if A.max() > 0:
        A = A / A.max()

    # Peak and trough detection
    is_peak = np.concatenate(([False], (A[1:-1] > A[:-2]) & (A[1:-1] > A[2:]), [False]))
    peaks = np.where(is_peak & (A > min_peak))[0]
    is_trough = np.concatenate(([False], (A[1:-1] < A[:-2]) & (A[1:-1] < A[2:]), [False]))
    troughs = np.where(is_trough)[0]

    # Build syllable segments
    syllables = []
    for i in range(len(troughs) - 1):
        s_idx = troughs[i]
        e_idx = troughs[i+1]
        # Discard short segments.
        duration = t[e_idx] - t[s_idx]
        # print(t[s_idx], t[e_idx], duration)
        if duration < 0.025:  # skip segments shorter than 25 ms
            #print(f"Skipping segment due to short duration: {duration:.3f}s")
            continue
        cands = [p for p in peaks if p > s_idx and p < e_idx]
        if cands:
            vals = A[cands]
            peak_idx = cands[np.argmax(vals)]
            syllables.append((t[s_idx], t[peak_idx], t[e_idx]))

    # Optional plotting
    if show_plots:
        plt.figure(figsize=(12, 4))
        plt.plot(t, A, label='Envelope')
        plt.plot(t[peaks], A[peaks], 'ro', label='Peaks')
        plt.plot(t[troughs], A[troughs], 'go', label='Troughs')
        for (s, p, e) in syllables:
            plt.axvspan(s, e, color='orange', alpha=0.3)
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Amplitude')
        plt.legend()
        plt.show()

    return syllables, t, A


def segment_audio(audio_file: str,
                  samplerate: int = 16000,
                  **kwargs) -> tuple:
    """
    Load audio file and segment via segment_waveform.
    Returns (syllables, t, A).
    """
    audio, sr = librosa.load(audio_file, sr=samplerate)
    return segment_waveform(audio, sr, **kwargs)


def parse_textgrid(textgrid_file: str, tier_name: str = 'speaker : syllables') -> list:
    tg = textgrid.openTextgrid(textgrid_file, includeEmptyIntervals=True)
    tier = tg.getTier(tier_name)
    sylls = []
    for start, end, label in tier.entries:
        if label.strip():
            sylls.append((start, end, label))
    return sylls


def is_overlap_above_threshold(pred, gt, threshold: float) -> bool:
    p0, p1 = pred
    g0, g1 = gt[0], gt[1]
    inter = max(0, min(p1, g1) - max(p0, g0))
    union = max(p1, g1) - min(p0, g0)
    return (inter / union) > threshold if union > 0 else False


def evaluate_predictions(gt_sylls, pred_sylls, syllables, method='peak-in-region', threshold=0.7) -> dict:
    TP, FP = 0, 0
    matched = set()
    for i, pred in enumerate(pred_sylls):
        found = False
        for j, gt in enumerate(gt_sylls):
            if j in matched: continue
            if method=='peak-in-region':
                if gt[0] <= syllables[i][1] <= gt[1]: found = True
            else:
                if is_overlap_above_threshold(pred, gt, threshold): found = True
            if found:
                TP += 1; matched.add(j); break
        if not found: FP += 1
    FN = len(gt_sylls) - len(matched)
    return {'TP':TP, 'FP':FP, 'FN':FN}


def evaluate_segmentation(audio_files, textgrid_files, method='peak-in-region', threshold_overlap=0.7, **kwargs) -> tuple:
    tot_TP = tot_FP = tot_FN = 0
    for af, tg in zip(audio_files, textgrid_files):
        sylls, t, A = segment_audio(af, **kwargs)
        preds = [(s,e) for (s,p,e) in sylls]
        gt = parse_textgrid(tg)
        res = evaluate_predictions(gt, preds, sylls, method, threshold_overlap)
        tot_TP += res['TP']; tot_FP += res['FP']; tot_FN += res['FN']
    prec = tot_TP/(tot_TP+tot_FP) if (tot_TP+tot_FP)>0 else 0
    rec  = tot_TP/(tot_TP+tot_FN) if (tot_TP+tot_FN)>0 else 0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
    return prec, rec, f1


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Syllable segmentation and evaluation')
    sub = parser.add_subparsers(dest='cmd')
    p1 = sub.add_parser('segment')
    p1.add_argument('--audio', required=True)
    p1.add_argument('--show_plots', action='store_true')
    p2 = sub.add_parser('evaluate')
    p2.add_argument('--audio_dir', required=True)
    p2.add_argument('--textgrid_dir', required=True)
    p2.add_argument('--method', choices=['peak-in-region','overlap'], default='peak-in-region')
    p2.add_argument('--threshold_overlap', type=float, default=0.7)
    args = parser.parse_args()
    if args.cmd=='segment':
        syls, t, A = segment_audio(args.audio, show_plots=args.show_plots)
        for s,p,e in syls:
            print(f"{s:.3f}-{p:.3f}-{e:.3f}")
    elif args.cmd=='evaluate':
        import glob
        afs = sorted(glob.glob(os.path.join(args.audio_dir, '*.wav')) + glob.glob(os.path.join(args.audio_dir, '*.flac')))
        tgs = sorted(glob.glob(os.path.join(args.textgrid_dir, '*.TextGrid')) + glob.glob(os.path.join(args.textgrid_dir, '*.textgrid')))
        prec, rec, f1 = evaluate_segmentation(afs, tgs, method=args.method, threshold_overlap=args.threshold_overlap)
        print(f"Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")

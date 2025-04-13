#!/usr/bin/env python
"""
findsylls.py

This module implements syllable segmentation using an amplitude-spectrum based method 
and provides routines to evaluate segmentation against aligned TextGrid syllable annotations.

The method is as follows:
    1. Compute the amplitude spectrum in a 16 msec window 100 times per second.
    2. At each time step, subtract the sum of the frequencies above 3 kHz from the sum of the frequencies below 3 kHz.
    3. Set negative values to zero.
    4. Smooth the resulting time function by convolving with a 70-msec Hamming window.
    5. Select all peaks whose value is greater than 4% of the maximum value.
    6. Create syllable segments by grouping adjacent troughs (valleys) and assigning the highest peak between them.
    
Dependencies:
    - numpy
    - matplotlib
    - scipy
    - librosa
    - textgrid (install via `pip install textgrid`)

Usage:
    As a module (example):
      from findsylls import segment_audio, evaluate_segmentation
      syllables, t, A = segment_audio("path/to/audio.wav", show_plots=True)
      print(syllables)
      
      # For corpus-level evaluation:
      audio_list = ["audio1.wav", "audio2.wav", ...]
      tg_list = ["tg1.TextGrid", "tg2.TextGrid", ...]
      precision, recall, f1 = evaluate_segmentation(audio_list, tg_list, method="peak-in-region")
      print(precision, recall, f1)
      
    From the command line:
      For segmentation:
        $ python findsylls.py segment --audio path/to/audio.wav [--show_plots]
      For evaluation:
        $ python findsylls.py evaluate --audio_dir path/to/audio_dir --textgrid_dir path/to/tg_dir
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


def segment_audio(audio_file, samplerate=16000, nfft=256, window_length=256, 
                  step=160, min_peak=0.04, smoothing_window_samples=7,
                  pivot_freq=3000, show_plots=False):
    """
    Segment an audio file into syllables using the amplitude-spectrum based method.
    
    Parameters:
        audio_file (str): Path to the audio file.
        samplerate (int): Sampling rate to use (assumes audio is in Hz).
        nfft (int): Number of FFT points (default: 256 → 16 msec at 16 kHz).
        window_length (int): Window length in samples (default: 256).
        step (int): Step size in samples for analysis (default: 160 for 10 msec steps).
        min_peak (float): Minimum normalized amplitude (4% of max) required for a peak.
        smoothing_window_samples (int): Number of samples for the Hamming smoothing window (7 samples ≈ 70 msec).
        pivot_freq (float): Frequency (in Hz) to separate the low and high frequency sums (default: 3000 Hz).
        show_plots (bool): If True, display plots of the amplitude envelope, peaks, troughs, and syllable segments.
        
    Returns:
        syllables (list of tuple): List of syllable segments as (start_time, peak_time, end_time) in seconds.
        t (np.array): Time vector corresponding to the amplitude envelope.
        A (np.array): Normalized amplitude envelope.
    """
    # Load audio (resampling if necessary)
    audio, sr = librosa.load(audio_file, sr=samplerate)
    
    # Calculate spectrogram using a Hamming window.
    noverlap = window_length - step  # overlap between consecutive windows
    f, t, Sxx = spectrogram(audio, fs=sr, nfft=nfft, window='hamming', 
                            nperseg=window_length, noverlap=noverlap, mode='magnitude')
    # Adjust the time vector to center the analysis window.
    t = t + (nfft / sr) / 2

    # Determine pivot bin corresponding to pivot_freq.
    freq_resolution = sr / nfft  # Hz per bin
    pivot_bin = int(np.ceil(pivot_freq / freq_resolution))
    
    # Sum energy for frequencies below pivot (ignoring the DC component, i.e., starting at index 1)
    Sum1 = np.sum(Sxx[1:pivot_bin, :], axis=0)
    # Sum energy for frequencies above pivot up to Nyquist (nfft//2)
    Sum2 = np.sum(Sxx[pivot_bin:(nfft//2), :], axis=0)
    
    # Compute the difference (low minus high) and clip negatives.
    diff_signal = Sum1 - Sum2
    diff_signal[diff_signal < 0] = 0
    
    # Smooth the resulting signal using a Hamming window filter (zero-phase filtering).
    smooth_filter = hamming(smoothing_window_samples)
    smooth_signal = filtfilt(smooth_filter, [1.0], diff_signal)
    
    print(smooth_signal)
    # Normalize the envelope so that its maximum is 1.
    if np.max(smooth_signal) == 0:
        A = smooth_signal
    else:
        A = smooth_signal / np.max(smooth_signal)
    
    # --- Peak Detection ---
    # Identify local peaks: a point is a peak if it is greater than its immediate neighbors.
    is_peak = np.concatenate(([False], (A[1:-1] > A[:-2]) & (A[1:-1] > A[2:]), [False]))
    # Only consider peaks above the minimum threshold.
    valid_peaks = is_peak & (A > min_peak)
    peak_indices = np.where(valid_peaks)[0]
    
    # --- Trough Detection ---
    # Identify local troughs (valleys): a point is a trough if it is lower than its immediate neighbors.
    is_trough = np.concatenate(([False], (A[1:-1] < A[:-2]) & (A[1:-1] < A[2:]), [False]))
    trough_indices = np.where(is_trough)[0]
    
    # --- Syllable Segmentation ---
    # For each pair of consecutive troughs, if there is at least one peak between them,
    # form a syllable segment: (trough_start, highest_peak between, trough_end)
    syllables = []
    for i in range(len(trough_indices) - 1):
        start_idx = trough_indices[i]
        end_idx = trough_indices[i + 1]
        # Find peaks between the two trough indices
        candidate_peaks = [p for p in peak_indices if p > start_idx and p < end_idx]
        if candidate_peaks:
            # Pick the peak with the highest amplitude value
            candidate_values = A[candidate_peaks]
            peak_idx = candidate_peaks[np.argmax(candidate_values)]
            syllables.append((t[start_idx], t[peak_idx], t[end_idx]))
    
    if show_plots:
        plt.figure(figsize=(12, 4))
        plt.plot(t, A, label="Normalized Amplitude Envelope")
        plt.plot(t[peak_indices], A[peak_indices], 'ro', label="Detected Peaks")
        plt.plot(t[trough_indices], A[trough_indices], 'go', label="Detected Troughs")
        for start, peak, end in syllables:
            plt.axvspan(start, end, color='orange', alpha=0.3)
        plt.xlabel("Time (s)")
        plt.ylabel("Normalized Amplitude")
        plt.title("Amplitude Envelope with Detected Peaks, Troughs, and Syllable Segments")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return syllables, t, A


def parse_textgrid(textgrid_file, tier_name="speaker : syllables"):
    """
    Parse a Praat TextGrid file to extract ground truth syllable intervals.
    
    Parameters:
        textgrid_file (str): Path to the TextGrid file.
        tier_name (str): Name of the tier to look for syllable annotations.
        
    Returns:
        sylls (list of tuple): A list of tuples (start_time, end_time, label) for each annotated syllable.
    """
    tg = textgrid.openTextgrid(textgrid_file, includeEmptyIntervals=True)
    tier = tg.getTier(tier_name)
    sylls = []
    for entry in tier.entries:
        start, end, label = entry
        if label.strip():
            sylls.append((start, end, label))
    return sylls


def is_overlap_above_threshold(pred_interval, gt_interval, threshold):
    """
    Check if the overlap between the predicted interval and ground truth interval is above a given threshold.
    
    Parameters:
        pred_interval (tuple): (start, end) of the predicted syllable.
        gt_interval (tuple): (start, end, label) for the ground truth syllable.
        threshold (float): Overlap ratio threshold (intersection over union).
        
    Returns:
        bool: True if the overlap exceeds the threshold, otherwise False.
    """
    p_start, p_end = pred_interval
    g_start, g_end = gt_interval[0], gt_interval[1]
    inter = max(0, min(p_end, g_end) - max(p_start, g_start))
    union = max(p_end, g_end) - min(p_start, g_start)
    if union == 0:
        return False
    return (inter / union) > threshold


def evaluate_predictions(gt_sylls, pred_sylls, syllables, method="peak-in-region", threshold=0.7):
    """
    Evaluate the segmentation predictions against ground truth.
    
    Two evaluation strategies are supported:
      - "peak-in-region": A predicted syllable is a true positive if its peak lies within a ground truth interval.
      - "overlap": A predicted syllable (start to end) is matched to a ground truth interval if the overlap ratio is above a threshold.
      
    Parameters:
        gt_sylls (list): Ground truth syllable intervals (start, end, label).
        pred_sylls (list): Predicted syllable intervals as (start, end) extracted from segments.
        syllables (list): Predicted syllable segments as (start, peak, end).
        method (str): Matching method ("peak-in-region" or "overlap").
        threshold (float): Overlap threshold (used if method=="overlap").
        
    Returns:
        dict: A dictionary with counts for "TP" (true positives), "FP" (false positives), and "FN" (false negatives).
    """
    TP = 0
    FP = 0
    matched_gt = set()
    
    for i, pred in enumerate(pred_sylls):
        match_found = False
        for j, gt in enumerate(gt_sylls):
            if j in matched_gt:
                continue
            if method == "peak-in-region":
                # Check whether the predicted peak (from syllables[i]) is within the ground truth interval.
                if gt[0] <= syllables[i][1] <= gt[1]:
                    match_found = True
            elif method == "overlap":
                if is_overlap_above_threshold(pred, gt, threshold):
                    match_found = True
            if match_found:
                TP += 1
                matched_gt.add(j)
                break
        if not match_found:
            FP += 1
    FN = len(gt_sylls) - len(matched_gt)
    return {"TP": TP, "FP": FP, "FN": FN}


def evaluate_segmentation(audio_files, textgrid_files, method="peak-in-region", threshold_overlap=0.7, **segmentation_kwargs):
    """
    Evaluate segmentation performance over a corpus of audio and TextGrid files.
    
    For each file, the function:
      1. Performs segmentation using the provided method.
      2. Parses ground truth syllable intervals from the corresponding TextGrid.
      3. Evaluates predictions using the selected matching strategy.
      
    Parameters:
        audio_files (list of str): List of paths to audio files.
        textgrid_files (list of str): List of paths to corresponding TextGrid files.
        method (str): Evaluation method: "peak-in-region" or "overlap".
        threshold_overlap (float): Overlap threshold (for "overlap" method).
        segmentation_kwargs: Extra keyword arguments passed to segment_audio().
        
    Returns:
        precision (float), recall (float), f1 (float): Aggregated evaluation metrics.
    """
    total_TP = 0
    total_FP = 0
    total_FN = 0
    
    for audio_file, tg_file in zip(audio_files, textgrid_files):
        syllables, t, A = segment_audio(audio_file, **segmentation_kwargs)
        # For evaluation, use the predicted segment boundaries (start and end from each syllable)
        pred_intervals = [(start, end) for (start, peak, end) in syllables]
        gt_sylls = parse_textgrid(tg_file)
        results = evaluate_predictions(gt_sylls, pred_intervals, syllables, method=method, threshold=threshold_overlap)
        total_TP += results["TP"]
        total_FP += results["FP"]
        total_FN += results["FN"]
    
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# -------------------- Command Line Interface --------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Syllable segmentation using an amplitude-spectrum based method and evaluation."
    )
    subparsers = parser.add_subparsers(dest="command", help="Command: 'segment' or 'evaluate'")
    
    # Command for segmenting a single file.
    parser_seg = subparsers.add_parser("segment", help="Segment a single audio file.")
    parser_seg.add_argument("--audio", required=True, help="Path to the audio file to segment.")
    parser_seg.add_argument("--show_plots", action="store_true", help="Display plots for debugging.")
    
    # Command for evaluating a corpus.
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate segmentation over a corpus.")
    parser_eval.add_argument("--audio_dir", required=True, help="Directory containing audio files (.wav or .flac).")
    parser_eval.add_argument("--textgrid_dir", required=True, help="Directory containing corresponding TextGrid files.")
    parser_eval.add_argument("--method", choices=["peak-in-region", "overlap"], default="peak-in-region",
                             help="Evaluation matching method (default: peak-in-region).")
    parser_eval.add_argument("--threshold_overlap", type=float, default=0.7,
                             help="Overlap threshold (if using 'overlap' evaluation).")
    
    args = parser.parse_args()
    
    if args.command == "segment":
        syllables, t, A = segment_audio(args.audio, show_plots=args.show_plots)
        print("Detected syllable segments (start, peak, end in seconds):")
        for seg in syllables:
            print(f"{seg[0]:.3f} - {seg[1]:.3f} - {seg[2]:.3f}")
    
    elif args.command == "evaluate":
        # Gather audio files (accepts .wav and .flac)
        audio_files = sorted(glob.glob(os.path.join(args.audio_dir, "*.wav")) +
                             glob.glob(os.path.join(args.audio_dir, "*.flac")))
        # Gather TextGrid files (both .TextGrid and .textgrid extensions)
        textgrid_files = sorted(glob.glob(os.path.join(args.textgrid_dir, "*.TextGrid")) +
                                glob.glob(os.path.join(args.textgrid_dir, "*.textgrid")))
        if len(audio_files) != len(textgrid_files):
            print("Warning: The number of audio files and TextGrid files does not match.")
        precision, recall, f1 = evaluate_segmentation(
            audio_files, textgrid_files, method=args.method,
            threshold_overlap=args.threshold_overlap
        )
        print("Corpus Evaluation Metrics:")
        print(f" Precision: {precision:.3f}")
        print(f" Recall:    {recall:.3f}")
        print(f" F1 Score:  {f1:.3f}")

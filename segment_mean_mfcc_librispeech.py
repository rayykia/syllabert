# segment_mean_mfcc_librispeech.py

import os
import glob
import json
import argparse
import numpy as np
import librosa
from tqdm import tqdm

from findsylls import segment_audio  # assumes your segmentation function is imported from findsylls.py

def process_librispeech_file(audio_file, output_dir, samplerate=16000, n_mfcc=13):
    base = os.path.splitext(os.path.basename(audio_file))[0]
    speaker_id = 0
    chapter_id = 0
    # speaker_id = audio_file.split("/")[-3]
    # chapter_id = audio_file.split("/")[-2]
    utterance_id = f"{speaker_id}-{chapter_id}-{base}"

    try:
        syllables, t, A = segment_audio(audio_file, samplerate=samplerate, show_plots=False)
        audio, sr = librosa.load(audio_file, sr=samplerate)
    except Exception as e:
        print(f"Skipping {audio_file} due to error: {e}")
        return []

    vectors_info = []
    mfcc_vectors = []

    for idx, (start, peak, end) in enumerate(syllables):
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        syll = audio[start_sample:end_sample]
        duration = (end - start)

        if len(syll) < int(0.025 * sr):  # skip segments shorter than 25 ms
            continue

        mfcc = librosa.feature.mfcc(y=syll, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = mfcc.mean(axis=1)  # shape: (n_mfcc,)

        mfcc_vectors.append(mfcc_mean)

        vectors_info.append({
            "audio_file": os.path.abspath(audio_file),
            "utterance_id": utterance_id,
            "segment_index": idx,
            "segment_start": float(start),
            "segment_end": float(end),
            "duration": duration,
            "vector_index": len(mfcc_vectors) - 1
        })

    return mfcc_vectors, vectors_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech_root", required=True, help="Path to LibriSpeech split root")
    parser.add_argument("--output_dir", required=True, help="Directory to save MFCC numpy files")
    parser.add_argument("--manifest_file", required=True, help="Path to output manifest file (JSONL)")
    parser.add_argument("--n_mfcc", type=int, default=13, help="Number of MFCC coefficients")
    parser.add_argument("--samplerate", type=int, default=16000, help="Sampling rate")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    flac_files = sorted(glob.glob(os.path.join(args.librispeech_root, "**/*.flac"), recursive=True))
    print(f"Found {len(flac_files)} .flac files in {args.librispeech_root}")

    all_vectors = []
    all_metadata = []

    for file in tqdm(flac_files, desc="Processing LibriSpeech"):
        result = process_librispeech_file(
            audio_file=file,
            output_dir=args.output_dir,
            samplerate=args.samplerate,
            n_mfcc=args.n_mfcc
        ) 
        if not result:
            continue
        vectors, metadata = result
        all_vectors.extend(vectors)
        all_metadata.extend(metadata)

    # Save all MFCC mean vectors in one file
    vectors_np = np.stack(all_vectors, axis=0)
    vectors_path = os.path.join(args.output_dir, "syllable_mfcc_vectors.npy")
    np.save(vectors_path, vectors_np)

    # Save manifest metadata
    manifest_path = args.manifest_file
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in all_metadata:
            f.write(json.dumps(entry) + "\n")

    print(f"Saved {len(all_vectors)} syllable MFCC vectors to {vectors_path}")
    print(f"Saved manifest with metadata to {manifest_path}")

if __name__ == "__main__":
    main()

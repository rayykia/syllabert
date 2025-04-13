# audiojack
CIS 6200 - Advanced Deep Learning final project

---
# SyllaBERT: Syllable-Aware HuBERT Pretraining

SyllaBERT is a syllable-aware variant of HuBERT that operates on raw audio and predicts discrete pseudo-labels over syllables rather than fixed frames. This enables more linguistically grounded modeling of spoken language with better data and parameter efficiency.

---

## 🧱 Pipeline Overview

1. **Syllable Segmentation:** Detect syllables in raw speech using a valley-peak-valley heuristic over a modulation-based amplitude envelope  
2. **MFCC Extraction:** Extract a single MFCC vector per syllable  
3. **Clustering:** Apply k-means to the MFCCs to generate pseudo-labels  
4. **Training:** Train a HuBERT-style model using raw audio input and masked syllable-label prediction

---

## ⚙️ Environment Setup

```bash
conda create -n syllabert python=3.11
conda activate syllabert

pip install torch torchaudio librosa tqdm scipy scikit-learn textgrid
```

---

## 🎧 1. Download LibriSpeech 100h Subset

```bash
mkdir -p ~/datasets/LibriSpeech
cd ~/datasets/LibriSpeech
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xvzf train-clean-100.tar.gz
```

---

## 🎙 2. Segment Audio and Extract MFCCs

```bash
python segment_mean_mfcc_librispeech.py \
  --librispeech_root ~/datasets/LibriSpeech/train-clean-100 \
  --output_dir ./data/syllabert_clean100/features \
  --manifest_file ./data/syllabert_clean100/features/manifest.jsonl
```

This script:
- Segments each utterance into syllables using an amplitude-based method
- Computes a single MFCC vector per syllable
- Saves `.npy` feature files and a `manifest.jsonl` describing them

---

## 🔀 3. Cluster Syllable MFCCs

```bash
python cluster_syllable_vectors.py \
  --manifest ./data/syllabert_clean100/features/manifest.jsonl \
  --output_dir ./data/syllabert_clean100/clustering \
  --n_clusters 100
```

This script:
- Loads all MFCC vectors from the manifest
- Applies k-means clustering with `n_clusters`
- Writes a new manifest with cluster IDs as pseudo-labels

Output:
```
./data/syllabert_clean100/clustering/labeled_manifest.jsonl
```

---

## 🚀 4. Train the SyllaBERT Model

```bash
python train_syllabert.py
```

This script:
- Loads raw waveforms and syllable boundary metadata
- Applies a convolutional frontend + transformer encoder
- Uses HuBERT-style masking (65% of syllables per batch)
- Computes loss only over masked syllables
- Logs batch accuracy and loss
- Saves model checkpoints after each epoch:
  ```
  checkpoints/syllabert_wave_epochN.pt
  checkpoints/syllabert_wave_latest.pt
  ```

---

## 📂 Directory Layout

```
data/
└── syllabert_clean100/
    ├── features/
    │   ├── <utterance_id>.npy          # MFCC vectors
    │   └── manifest.jsonl              # Metadata (audio paths, timings, features)
    └── clustering/
        ├── kmeans_100.pkl              # KMeans model
        └── labeled_manifest.jsonl      # MFCCs with cluster labels
```

---

## 📒 Notes

- Syllables are pooled from raw waveforms; targets are cluster IDs over syllable MFCCs.
- The architecture matches HuBERT, but the learning unit is a syllable.
- Masking and loss computation follow HuBERT’s 65% random span masking strategy — adapted for syllables.

---

## 📚 References

- Hsu et al., 2021. [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447)
```
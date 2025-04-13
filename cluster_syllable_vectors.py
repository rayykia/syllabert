#!/usr/bin/env python
"""
cluster_syllable_vectors.py

Clusters syllable-level MFCC vectors using k-means, producing pseudo-labels for HuBERT pretraining.

Input:
  - .npy file of MFCC vectors (N x 13)
  - .jsonl manifest from syllable segmentation

Output:
  - .pkl file of trained KMeans model
  - .npy file of cluster labels (N,)
  - .jsonl manifest with cluster IDs included
"""

import argparse
import json
import joblib
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors_file", required=True, help="Path to .npy file of syllable MFCC vectors")
    parser.add_argument("--manifest_file", required=True, help="Path to original manifest .jsonl")
    parser.add_argument("--output_dir", required=True, help="Directory to save k-means model and label outputs")
    parser.add_argument("--n_clusters", type=int, default=100, help="Number of clusters (default: 100)")
    args = parser.parse_args()

    # Load MFCC vectors
    print(f"Loading MFCC vectors from {args.vectors_file}...")
    vectors = np.load(args.vectors_file)  # shape: (N, 13)

    # Run k-means
    print(f"Running k-means with k={args.n_clusters} on {vectors.shape[0]} syllable vectors...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init="auto")
    cluster_ids = kmeans.fit_predict(vectors)

    # Save model and labels
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = f"{args.output_dir}/syllable_kmeans_k{args.n_clusters}.pkl"
    labels_path = f"{args.output_dir}/syllable_cluster_labels.npy"
    labeled_manifest_path = f"{args.output_dir}/labeled_manifest.jsonl"

    joblib.dump(kmeans, model_path)
    np.save(labels_path, cluster_ids)

    print(f"Saved k-means model to {model_path}")
    print(f"Saved cluster labels to {labels_path}")

    # Load original manifest and write new one with cluster IDs
    print("Writing labeled manifest...")
    with open(args.manifest_file, "r", encoding="utf-8") as fin, \
         open(labeled_manifest_path, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(tqdm(fin)):
            entry = json.loads(line)
            entry["cluster_id"] = int(cluster_ids[idx])
            fout.write(json.dumps(entry) + "\n")

    print(f"Labeled manifest written to {labeled_manifest_path}")

if __name__ == "__main__":
    import os
    main()

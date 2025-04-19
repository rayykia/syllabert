#!/usr/bin/env python
"""
train_syllabert.py

Fine‑tune SyllaBERTEncoder (imported as SyllaBERT) on syllable cluster targets.
"""
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from syllabert_model import SyllaBERT  # alias for SyllaBERTEncoder with projection
from syllable_dataset import SyllableDataset, collate_syllable_utterances


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
    device = get_device()
    # instantiate model: only hubert_pretrained_model arg
    model = SyllaBERT(
        hubert_pretrained_model=args.hubert_model
    ).to(device)

    # prepare data
    dataset = SyllableDataset(
        manifest_path=args.manifest,
        samplerate=args.sr
    )
    loader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=True,
        collate_fn=collate_syllable_utterances,
        pin_memory=True
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, steps = 0.0, 0
        for inputs, targets_list, segments in loader:
            if inputs is None:
                continue
            inputs = inputs.to(device)
            targets = torch.cat(targets_list).to(device)
            # forward returns list of (N_syll, num_classes) logits
            logits_list = model(inputs, args.sr)
            # flatten and compute loss
            all_logits = torch.cat(logits_list, dim=0)
            loss = F.cross_entropy(all_logits, targets, ignore_index=-100)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / steps if steps else 0.0
        print(f"Epoch {epoch}/{args.epochs}, Loss: {avg_loss:.4f}")

        os.makedirs(args.out_dir, exist_ok=True)
        ckpt = os.path.join(args.out_dir, f"syllabert_epoch{epoch}.pt")
        model.save_checkpoint(ckpt)
        model.save_checkpoint(os.path.join(args.out_dir, "syllabert_latest.pt"))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--hubert_model', type=str, default='facebook/hubert-base-ls960')
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--out_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    train(args)

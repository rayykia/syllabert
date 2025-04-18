import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from boundary_classifier import HubertSyllableBoundary
from syllable_dataset import SyllableDataset, collate_syllable_utterances


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def make_frame_labels(segments, num_frames):
    labels = torch.zeros(num_frames, dtype=torch.long)
    for (s, e) in segments:
        if 0 <= s < num_frames:
            labels[s] = 1
    return labels


def train_boundary_classifier(
    manifest_path: str,
    sampling_rate: int = 16000,
    batch_size: int = 4,
    lr: float = 1e-5,
    num_epochs: int = 5,
):
    device = get_device()
    model = HubertSyllableBoundary().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    dataset = SyllableDataset(manifest_path=manifest_path, samplerate=sampling_rate)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_syllable_utterances,
        pin_memory=True,
    )

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for waveforms, labels_list, segments in tqdm(loader, desc=f"Epoch {epoch}"):
            # waveforms: (B, 1, T_samples)
            B, _, T = waveforms.shape
            waveforms = waveforms.squeeze(1).to(device)  # (B, T)

            # Forward: returns (loss, logits) in training, logits in eval
            output = model(waveforms, sampling_rate)
            if isinstance(output, tuple):
                loss, logits = output
            else:
                logits = output
                loss = None

            _, T_frames, _ = logits.shape

            # If segmentation-only pseudo-label mode, labels_list unused
            # Otherwise, can use labels_list for supervised labels

            # Backprop only during training
            if model.training and loss is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        if n_batches > 0:
            avg_loss = total_loss / n_batches
            print(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch}: No training steps taken.")

    torch.save(model.state_dict(), "boundary_classifier_selfsupervised.pt")
    print("Model saved to boundary_classifier_selfsupervised.pt")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train HuBERT boundary classifier.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    train_boundary_classifier(
        manifest_path=args.manifest,
        sampling_rate=args.sr,
        batch_size=args.bs,
        lr=args.lr,
        num_epochs=args.epochs,
    )

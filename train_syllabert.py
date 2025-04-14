# train_syllabert.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from syllabert_model import SyllaBERT
from syllable_dataset import SyllableDataset, collate_syllable_utterances
from tqdm import tqdm
import logging

# Set up basic logging configuration.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train():
    torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection.
    device = get_device()

    model = SyllaBERT(input_dim=1, embed_dim=768, num_layers=12, num_heads=12, num_classes=100)
    model = model.to(device)

    dataset = SyllableDataset(
        manifest_path="./data/syllabert_clean100/clustering/labeled_manifest.jsonl",
        samplerate=16000
    )
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_syllable_utterances,
        pin_memory=True
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Parameters for masking.
    mask_prob = 0.65  # 65% of syllable segments will be masked.
    default_mask_length = 10  # Ignored if syllable segments are provided.

    num_epochs = 10
    log_interval = 10  # Log every 10 batches.

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        batch_count = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, (inputs, targets, segments) in enumerate(progress_bar, start=1):
            if inputs is None:
                continue

            # Move inputs and targets to device.
            inputs = inputs.to(device)
            targets = torch.cat(targets).to(device)

            # Forward pass with self-supervised masking on the convolutional features.
            logits, mask_flags = model(
                inputs,
                syllable_segments=segments,
                apply_mask=True,
                mask_prob=mask_prob,
                mask_length=default_mask_length
            )
            loss = model.compute_loss(logits, targets, mask_flags=mask_flags)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            if batch_idx % log_interval == 0:
                avg_loss = total_loss / batch_count
                logger.info(f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}], Avg Loss: {avg_loss:.4f}")

            progress_bar.set_postfix(loss=loss.item())

        epoch_avg_loss = total_loss / batch_count if batch_count else float("inf")
        logger.info(f"End of Epoch {epoch}: Average Loss: {epoch_avg_loss:.4f}")
        model.save_checkpoint(f"checkpoints/syllabert_wave_epoch{epoch}.pt")
        torch.save(model.state_dict(), "checkpoints/syllabert_wave_latest.pt")


if __name__ == "__main__":
    train()

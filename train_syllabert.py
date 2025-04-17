import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from syllabert_model import SyllaBERT
from syllable_dataset import SyllableDataset, collate_syllable_utterances
from tqdm import tqdm
import logging

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
    device = get_device()
    model = SyllaBERT(
        input_dim=1,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        num_classes=100,
        dropout=0.1,
        max_syllables=200,
        sampling_rate=16000
    ).to(device)

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
    mask_prob = 0.65
    num_epochs = 10
    log_interval = 10

    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0.0

        for batch_idx, (waves, labels, pad_mask_data) in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch}"), start=1):

            if waves is None:
                continue

            # Move data to device
            waves = waves.to(device)       # [B,1,T]
            labels = labels.to(device)     # [B, S_data]
            pad_mask_data = pad_mask_data.to(device)  # [B, S_data]

            # Forward: segmentation, pooling, masking inside model
            logits, mask_tokens, pad_mask_model = model(
                waves,
                apply_mask=True,
                mask_prob=mask_prob
            )
            # logits: [B, S_model, C]
            B, S_model, _ = logits.size()

            # Build target tensor aligned to model's token count
            target_tokens = torch.full((B, S_model), -100, dtype=torch.long, device=device)
            for b in range(B):
                n_valid = (labels[b] != -100).sum().item()
                n_valid = min(n_valid, S_model)
                if n_valid > 0:
                    target_tokens[b, :n_valid] = labels[b, :n_valid]

            # Compute masked prediction loss
            loss = model.compute_loss(logits, target_tokens, mask_tokens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % log_interval == 0:
                avg_loss = total_loss / batch_idx
                logger.info(
                    f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}] Avg Loss: {avg_loss:.4f}"
                )

        epoch_avg = total_loss / len(dataloader)
        logger.info(f"End Epoch {epoch} Avg Loss {epoch_avg:.4f}")
        model.save_checkpoint(f"checkpoints/syllabert_epoch{epoch}.pt")
        torch.save(model.state_dict(), "checkpoints/syllabert_latest.pt")

if __name__ == "__main__":
    train()

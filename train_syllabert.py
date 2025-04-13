# train_syllabert.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from syllabert_model import SyllaBERT
from syllable_dataset import SyllableDataset, collate_syllable_utterances

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def train():
    device = get_device()

    model = SyllaBERT(input_dim=1, embed_dim=768, num_layers=12, num_heads=12, num_classes=100)
    model = model.to(device)

    dataset = SyllableDataset(
        manifest_path="./data/syllabert_clean100/clustering/labeled_manifest.jsonl",
        samplerate=16000
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_syllable_utterances)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    from tqdm import tqdm

    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, targets, segments) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            if inputs is None:
                continue

            inputs = inputs.to(device)
            # HuBERT-style masking: mask ~65% of syllables
            all_targets = torch.cat(targets)
            num_total = all_targets.shape[0]
            mask = torch.rand(num_total, device=device) < 0.65
            masked_targets = all_targets.clone()
            masked_targets[~mask] = -100  # ignore unmasked syllables
            targets = masked_targets.to(device)  # flatten all syllable targets

            logits = model(inputs, syllable_segments=segments)
            loss = model.compute_loss(logits, targets)
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                mask_indices = targets != -100
                correct = (preds[mask_indices] == targets[mask_indices]).sum().item()
                total = mask_indices.sum().item()
                acc = correct / total if total > 0 else 0.0
                #print(f"Batch {batch_idx + 1}: Loss = {loss.item():.4f}, Accuracy = {acc * 100:.2f}% ({correct}/{total})")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        model.save_checkpoint(f"checkpoints/syllabert_wave_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), "checkpoints/syllabert_wave_latest.pt")

if __name__ == "__main__":
    train()

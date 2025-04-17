import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from syllabert_model import SyllaBERT
from syllable_dataset import SyllableDataset, collate_syllable_utterances
from tqdm import tqdm
import argparse
from datetime import datetime
from loguru import logger
import os
import re

import warnings
warnings.filterwarnings("ignore")

now = datetime.now().strftime("%Y-%m-%d_%H-%M")

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), level="INFO")
os.makedirs('./logs', exist_ok=True)
logger.add(f"logs/train_{now}.log", level="INFO", format="{time} | {level} | {message}")




def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    

def load_latest_checkpoint(checkpoint_dir):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if re.match(r'syllabert_epoch(\d+)\.pt', f)]
    epochs = [int(re.search(r'epoch(\d+)', f).group(1)) for f in checkpoint_files]
    latest_epoch = max(epochs)
    latest_ckpt = f"syllabert_epoch{latest_epoch}.pt"
    path = os.path.join(checkpoint_dir, latest_ckpt)
    return path, latest_epoch


def train(args):
    torch.autograd.set_detect_anomaly(True)
    device = get_device()

    input_dim=1
    embed_dim=768
    num_layers=12
    num_heads=12
    num_classes=100
    logger.info(f"Loading model: {input_dim=}, {embed_dim=}, {num_layers=}, {num_heads=}, {num_classes=}.")
    model = SyllaBERT(input_dim, embed_dim, num_layers, num_heads, num_classes)

    epoch_n = 0
    if args.c:
        path, epoch_n = load_latest_checkpoint('./checkpoints/')
        model.load_state_dict(torch.load(path))
        logger.info(f"Loaded checkpoint: {path}")
    if args.continue_path is not None:
        epoch_n = int(re.search(r'epoch(\d+)', args.continue_path).group(1))
        model.load_state_dict(torch.load(args.continue_path))

    model = model.to(device)


    dataset = SyllableDataset(
        manifest_path="./data/syllabert_clean100/clustering/labeled_manifest.jsonl",
        samplerate=16000
    )
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_syllable_utterances,
        pin_memory=True
    )
    num_batches = len(dataloader)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    mask_prob = 0.65
    num_epochs = 35
    log_interval = 100


    for epoch in range(1+epoch_n, num_epochs+1 + epoch_n):
        model.train()
        total_loss = 0.0
        batch_count = 0
        batch_loss = 0.

        logger.info(f'Epoch {epoch}')

        with tqdm(total=num_batches, desc=f"Epoch {epoch}", dynamic_ncols=True, leave=False) as pbar:
            # for batch_idx, (inputs, syllable_labels, segments) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}"), start=1):
            for batch_idx, (inputs, syllable_labels, segments) in enumerate(dataloader, start=1):

                if inputs is None:
                    continue

                inputs = inputs.to(device)

                # Determine feature length via encoder (no masking)
                with torch.no_grad():
                    feats = model.encoder(inputs)  # [B, T_feat, C]
                B, T_feat, _ = feats.shape

                # Build frame-level targets and boundary flags of shape [B, T_feat]
                boundary_targets = torch.zeros((B, T_feat), device=device)
                frame_targets = torch.full((B, T_feat), -100, dtype=torch.long, device=device)

                for b, (segs, labs) in enumerate(zip(segments, syllable_labels)):
                    for (s, e), cid in zip(segs, labs.tolist()):
                        # clamp to feature length
                        s_clamped = max(0, min(s, T_feat))
                        e_clamped = max(s_clamped+1, min(e, T_feat))
                        boundary_targets[b, e_clamped-1] = 1
                        frame_targets[b, s_clamped:e_clamped] = cid

                # Forward pass with syllable-level masking
                logits, mask_idx, boundary_prob = model(
                    inputs,
                    boundary_targets=boundary_targets,
                    apply_mask=True,
                    mask_prob=mask_prob
                )

                # Compute losses
                cluster_loss = model.compute_cluster_loss(logits, frame_targets, mask_idx)
                boundary_loss = model.compute_boundary_loss(boundary_prob, boundary_targets)
                loss = cluster_loss + boundary_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

                # if batch_idx % log_interval == 0:
                #     avg_loss = total_loss / batch_count
                #     logger.info(
                #         f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}], Avg Loss: {avg_loss:.4f}"
                #     )

                with torch.no_grad():
                    total_loss += loss.item()
                    batch_count += 1
                    batch_loss += loss.item()
                    # if batch_idx % log_interval == 0:

                    #     if batch_idx != 0:
                    #         avg_batch_loss = batch_loss / log_interval
                            
                    #     else:
                    #         avg_batch_loss = loss.item()
                    #     logger.info(f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}], Avg Loss: {avg_batch_loss:.4f}")
                    #     batch_loss = 0.0
                    
                    preds = logits.argmax(dim=-1)
                    masked_preds = preds[mask_idx]
                    masked_targets = frame_targets[mask_idx]
                    correct = (masked_preds == masked_targets).sum().item()
                    total = mask_idx.sum().item()
                    acc = correct / total if total > 0 else 0.0

                    if batch_idx % log_interval == 0:
                        if batch_idx != 0:
                            avg_batch_loss = batch_loss / log_interval
                        else:
                            avg_batch_loss = loss.item()
                        logger.info(
                            f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}], "
                            f"Avg Loss: {avg_batch_loss:.4f}, "
                            f"Masked Acc: {acc * 100:.2f}% ({correct}/{total})"
                        )
                        batch_loss = 0.0
                pbar.update(1)

        epoch_avg = total_loss / batch_count if batch_count else float('inf')
        logger.info(f"End Epoch {epoch}: Avg Loss: {epoch_avg:.4f}")
        model.save_checkpoint(f"checkpoints/syllabert_epoch{epoch}.pt")
        torch.save(model.state_dict(), "checkpoints/syllabert_latest.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--continue', dest='continue_path', required=False, help='path to the model parameters to continue training')
    parser.add_argument('-c', required=False, action='store_true',help = 'continue training form the latest model')
    args = parser.parse_args()

    train(args)

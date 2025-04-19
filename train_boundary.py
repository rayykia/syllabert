import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger


from boundary_classifier import HubertSyllableBoundary
from syllable_dataset import SyllableDataset, collate_syllable_utterances

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), level="INFO")


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


    # if args.c:
    #     path, epoch_n = load_latest_checkpoint('./checkpoints/')
    #     model.load_state_dict(torch.load(path))
    #     logger.info(f"Loaded checkpoint: {path}")
    # if args.continue_path is not None:
    #     epoch_n = int(re.search(r'epoch(\d+)', args.continue_path).group(1))
    #     model.load_state_dict(torch.load(args.continue_path))
    #     logger.info(f"Loaded checkpoint: {args.continue_path}")


    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    dataset = SyllableDataset(manifest_path=manifest_path, samplerate=sampling_rate)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_syllable_utterances,
        pin_memory=True,
    )



    log_interval = 100

    num_batches = len(loader)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        batch_loss = 0.

        with tqdm(total=num_batches, desc=f"Epoch {epoch}", dynamic_ncols=True, leave=False) as pbar:
            for batch_idx, (waveforms, labels_list, segments) in enumerate(loader):
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
                    n_batches += 1

                    with torch.no_grad():
                        total_loss += loss.item()
                        batch_loss += loss.item()
                        if batch_idx % log_interval == 0:
                            if batch_idx != 0:
                                avg_batch_loss = batch_loss / log_interval
                            else:
                                avg_batch_loss = loss.item()
                            logger.info(
                                f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}], "
                                f"Avg Loss: {avg_batch_loss:.4f}, "
                            )
                            batch_loss = 0.0
                pbar.update(1)


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

    # parser.add_argument('--continue', dest='continue_path', required=False, help='path to the model parameters to continue training')
    # parser.add_argument('-c', required=False, action='store_true',help = 'continue training form the latest model')
    args = parser.parse_args()

    train_boundary_classifier(
        manifest_path=args.manifest,
        sampling_rate=args.sr,
        batch_size=args.bs,
        lr=args.lr,
        num_epochs=args.epochs,
    )

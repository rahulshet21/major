import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import EventPatchDataset, read_event_names
from src.evaluate import compute_metrics
from src.losses import BCEDiceLoss
from src.model import build_unet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def evaluate_model(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item()
            total_batches += 1

            probs = torch.sigmoid(logits).cpu().numpy()
            pred = (probs >= threshold).astype(np.uint8)
            target = masks.cpu().numpy().astype(np.uint8)

            all_preds.append(pred)
            all_targets.append(target)

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    flat_preds = preds.reshape(-1)
    flat_targets = targets.reshape(-1)

    metrics = compute_metrics(flat_preds, flat_targets, threshold=threshold)
    metrics["loss"] = total_loss / max(total_batches, 1)
    return metrics


def run_training(experiment, epochs=3, batch_size=4, learning_rate=1e-4, patch_size=256, overlap=64, seed=42, max_patches_per_event=None):
    set_seed(seed)
    device = get_device()

    if experiment == "s2":
        channels = 12
    elif experiment == "s1":
        channels = 3
    elif experiment == "fusion":
        channels = 30
    else:
        raise ValueError(f"Unsupported experiment: {experiment}")

    train_events = read_event_names("train_events.txt")
    val_events = read_event_names("val_events.txt")

    train_dataset = EventPatchDataset(
        train_events,
        experiment,
        patch_size=patch_size,
        overlap=overlap,
        max_patches_per_event=max_patches_per_event,
    )
    val_dataset = EventPatchDataset(
        val_events,
        experiment,
        patch_size=patch_size,
        overlap=overlap,
        max_patches_per_event=max_patches_per_event,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_unet(channels)
    model.to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    best_state = None
    history = []

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate_model(model, val_loader, criterion, device, threshold=0.5)
        scheduler.step(val_metrics["loss"])

        print(f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | val_dice={val_metrics['Dice']:.4f} | val_iou={val_metrics['IoU']:.4f}")

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["Dice"],
            "val_iou": val_metrics["IoU"],
            "val_precision": val_metrics["Precision"],
            "val_recall": val_metrics["Recall"],
            "val_f1": val_metrics["F1"],
        }
        history.append(record)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            checkpoint_path = CHECKPOINT_DIR / f"{experiment}_best.pth"
            torch.save(best_state, checkpoint_path)

    with open(LOG_DIR / f"{experiment}_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(LOG_DIR / f"{experiment}_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()) if history else [])
        writer.writeheader()
        writer.writerows(history)

    return model, history


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train wildfire segmentation U-Net")
    parser.add_argument("--experiment", choices=["s2", "s1", "fusion"], required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-patches-per-event", type=int, default=None)
    args = parser.parse_args()

    run_training(
        experiment=args.experiment,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patch_size=args.patch_size,
        overlap=args.overlap,
        seed=args.seed,
        max_patches_per_event=args.max_patches_per_event,
    )


if __name__ == "__main__":
    main()

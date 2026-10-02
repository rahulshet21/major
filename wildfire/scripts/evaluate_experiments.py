#!/usr/bin/env python3

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import aggregate_metrics, compute_metrics, event_model_input, save_event_visualization
from src.inference import sliding_window_predict
from src.model import build_unet

PROJECT_ROOT = ROOT
PROCESSED_DIR = PROJECT_ROOT / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
SPLITS_DIR = PROJECT_ROOT / "splits"


def load_event(event_name):
    image = np.load(PROCESSED_DIR / f"{event_name}_image.npy").astype(np.float32)
    mask = np.load(PROCESSED_DIR / f"{event_name}_mask.npy").astype(np.uint8)
    return image, mask


def evaluate_experiment(experiment, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    test_events = []
    test_path = SPLITS_DIR / "test_events.txt"
    with open(test_path, "r", encoding="utf-8") as f:
        test_events = [line.strip() for line in f if line.strip()]

    model = build_unet(in_channels=12 if experiment == "s2" else 3 if experiment == "s1" else 30)
    state = torch.load(CHECKPOINT_DIR / f"{experiment}_best.pth", map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    results = []
    all_event_metrics = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for event_name in test_events:
        image, gt = load_event(event_name)
        input_image = event_model_input(image, experiment)
        pred_prob = sliding_window_predict(model, input_image, patch_size=256, overlap=64, device=device)
        pred = (pred_prob >= threshold).astype(np.uint8)

        metrics = compute_metrics(pred, gt, threshold=threshold)
        all_event_metrics.append(metrics)
        results.append({
            "event": event_name,
            "IoU": metrics["IoU"],
            "Dice": metrics["Dice"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1"],
        })

        before_rgb = np.stack([
            input_image[0],
            input_image[1],
            input_image[2],
        ], axis=-1) if experiment in {"s2", "fusion"} else np.stack([
            image[12],
            image[13],
            image[14],
        ], axis=-1)

        after_rgb = np.stack([
            image[15],
            image[16],
            image[17],
        ], axis=-1) if experiment in {"fusion", "s2"} else np.stack([
            image[12],
            image[13],
            image[14],
        ], axis=-1)

        save_event_visualization(event_name, before_rgb, after_rgb, gt.astype(np.uint8), pred.astype(np.uint8), RESULTS_DIR)

    summary = aggregate_metrics(all_event_metrics)
    summary["experiment"] = experiment
    summary["n_test_events"] = len(test_events)
    summary["threshold"] = threshold

    out_path = RESULTS_DIR / f"{experiment}_metrics.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["event", "IoU", "Dice", "Precision", "Recall", "F1"])
        writer.writeheader()
        writer.writerows(results)

    print(f"=== {experiment.upper()} ===")
    print(summary)
    return summary, results


def main():
    for experiment in ["s2", "s1", "fusion"]:
        evaluate_experiment(experiment)


if __name__ == "__main__":
    main()

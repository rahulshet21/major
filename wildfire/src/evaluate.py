import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def compute_metrics(predictions, targets, threshold=0.5):
    predictions = predictions.astype(np.uint8)
    targets = targets.astype(np.uint8)

    tp = np.sum((predictions == 1) & (targets == 1))
    fp = np.sum((predictions == 1) & (targets == 0))
    fn = np.sum((predictions == 0) & (targets == 1))
    tn = np.sum((predictions == 0) & (targets == 0))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2.0 * tp / (2.0 * tp + fp + fn + 1e-8)

    return {
        "IoU": float(iou),
        "Dice": float(dice),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "threshold": float(threshold),
    }


def metric_from_logits(logits, targets, threshold=0.5):
    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = (probs >= threshold).astype(np.uint8)
    target = targets.astype(np.uint8)
    return compute_metrics(pred, target, threshold=threshold)


def event_model_input(image, experiment):
    if experiment == "s2":
        return image[:12]
    if experiment == "s1":
        return image[12:15]
    if experiment == "fusion":
        return image
    raise ValueError(f"Unsupported experiment: {experiment}")


def save_event_visualization(event_name, before_rgb, after_rgb, gt_mask, pred_mask, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(before_rgb)
    axes[0].set_title("Before S2 RGB")
    axes[0].axis("off")

    axes[1].imshow(after_rgb)
    axes[1].set_title("After S2 RGB")
    axes[1].axis("off")

    axes[2].imshow(gt_mask, cmap="gray")
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    axes[3].imshow(pred_mask, cmap="gray")
    axes[3].set_title("Prediction")
    axes[3].axis("off")

    overlay = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.float32)
    overlay[..., 0] = pred_mask.astype(np.float32) * 1.0
    overlay[..., 1] = gt_mask.astype(np.float32) * 0.8
    overlay[..., 2] = 0.2
    axes[4].imshow(overlay)
    axes[4].set_title("Overlay")
    axes[4].axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / f"{event_name}_prediction.png", dpi=200)
    plt.close(fig)


def aggregate_metrics(metric_list):
    keys = ["IoU", "Dice", "Precision", "Recall", "F1"]
    agg = {}
    for key in keys:
        agg[key] = float(np.mean([m[key] for m in metric_list])) if metric_list else 0.0
    return agg

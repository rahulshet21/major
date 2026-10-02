from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_before_after_rgb(image, experiment):
    if experiment == "s2":
        rgb = np.stack([image[2], image[1], image[0]], axis=-1)
        return rgb
    if experiment == "s1":
        rgb = np.stack([image[0], image[0], image[0]], axis=-1)
        return rgb
    if experiment == "fusion":
        if image.shape[0] == 30:
            rgb = np.stack([image[2], image[1], image[0]], axis=-1)
            return rgb
        if image.shape[0] >= 12:
            rgb = np.stack([image[2], image[1], image[0]], axis=-1)
            return rgb
    raise ValueError(f"Unsupported experiment: {experiment}")


def make_overlay(gt_mask, pred_mask):
    gt_mask = gt_mask.astype(np.float32)
    pred_mask = pred_mask.astype(np.float32)
    overlay = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.float32)
    overlay[..., 0] = pred_mask
    overlay[..., 1] = gt_mask * 0.7
    overlay[..., 2] = 0.1
    return overlay


def generate_event_visualization(event_name, before_image, after_image, gt_mask, pred_mask, output_dir, experiment="fusion"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    before_rgb = make_before_after_rgb(before_image, experiment)
    after_rgb = make_before_after_rgb(after_image, experiment)
    overlay = make_overlay(gt_mask, pred_mask)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
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

    axes[4].imshow(overlay)
    axes[4].set_title("Overlay")
    axes[4].axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / f"{event_name}_prediction.png", dpi=200)
    plt.close(fig)

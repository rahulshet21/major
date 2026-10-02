from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EVENT = "EMSR207_02LOUSA_02GRADING_MAP_v2_vector"
IMAGE_PATH = Path("processed") / f"{EVENT}_image.npy"
MASK_PATH = Path("processed") / f"{EVENT}_mask.npy"
PRED_PATH = Path("processed") / f"{EVENT}_fusion_prediction.npy"
OUTPUT_DIR = Path("processed22")


def normalize_display(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max - arr_min < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - arr_min) / (arr_max - arr_min + 1e-8)


def make_rgb_from_s2(image: np.ndarray, start: int = 0, end: int = 3) -> np.ndarray:
    bands = image[start:end]
    rgb = np.stack([bands[2], bands[1], bands[0]], axis=-1).astype(np.float32)
    rgb = normalize_display(rgb)
    return rgb


def make_overlay(gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    gt_mask = gt_mask.astype(np.float32)
    pred_mask = pred_mask.astype(np.float32)
    overlay = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.float32)
    overlay[..., 0] = pred_mask
    overlay[..., 1] = gt_mask * 0.7
    overlay[..., 2] = 0.1
    return overlay


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Missing image file: {IMAGE_PATH}")
    if not MASK_PATH.exists():
        raise FileNotFoundError(f"Missing mask file: {MASK_PATH}")
    if not PRED_PATH.exists():
        raise FileNotFoundError(f"Missing prediction file: {PRED_PATH}")

    image = np.load(IMAGE_PATH)
    gt_mask = np.load(MASK_PATH)
    pred_mask = np.load(PRED_PATH)

    if image.ndim != 3 or image.shape[0] != 30:
        raise ValueError(f"Expected image shape (30, H, W); got {image.shape}")
    if gt_mask.ndim != 2:
        raise ValueError(f"Expected mask shape (H, W); got {gt_mask.shape}")
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Prediction shape {pred_mask.shape} does not match mask {gt_mask.shape}")

    OUTPUT_DIR.mkdir(exist_ok=True)

    before_rgb = make_rgb_from_s2(image, 0, 3)
    after_rgb = make_rgb_from_s2(image, 15, 18)
    overlay = make_overlay(gt_mask, pred_mask)

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), constrained_layout=True)

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

    out_path = OUTPUT_DIR / f"{EVENT}_overlay.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Existing 30-channel input found: {IMAGE_PATH}")
    print(f"Shape: {image.shape}")
    print(f"Saved overlay figure to: {out_path}")


if __name__ == "__main__":
    main()

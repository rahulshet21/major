from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="Preview 30-channel original wildfire dataset .npy")
parser.add_argument("--input", type=str, required=True, help="Path to the 30-channel .npy file")
parser.add_argument("--out_dir", type=str, default="processed22", help="Directory to save the visualization")
args = parser.parse_args()

INPUT_PATH = Path(args.input)
OUTPUT_DIR = Path(args.out_dir)


def normalize_display(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max - arr_min < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - arr_min) / (arr_max - arr_min + 1e-8)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = normalize_display(arr)
    return np.clip(arr, 0.0, 1.0) * 255.0


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb_u8 = (rgb * 255.0).astype(np.uint8)
    plt.imsave(path, rgb_u8)


def save_grayscale(path: Path, channel: np.ndarray) -> None:
    channel = to_uint8(channel)
    plt.imsave(path, channel, cmap="gray")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Existing 30-channel input not found: {INPUT_PATH}")

    image = np.load(INPUT_PATH)
    print(f"Existing 30-channel input found: {INPUT_PATH}")
    print(f"Shape: {image.shape}")

    if image.ndim != 3:
        raise ValueError(f"Expected a 3D array (C, H, W), got ndim={image.ndim} with shape={image.shape}")
    if image.shape[0] != 30:
        raise ValueError(f"Expected 30 channels, got {image.shape[0]} channels. Shape={image.shape}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    
    input_name = INPUT_PATH.stem
    event_out_dir = OUTPUT_DIR / input_name
    event_out_dir.mkdir(parents=True, exist_ok=True)

    s2_before = image[0:3]
    s2_after = image[15:18]
    s1_before = image[12]
    s1_after = image[27]

    s2_before_rgb = np.transpose(s2_before, (1, 2, 0))
    s2_after_rgb = np.transpose(s2_after, (1, 2, 0))

    save_rgb(event_out_dir / "s2_before_rgb.png", s2_before_rgb)
    save_rgb(event_out_dir / "s2_after_rgb.png", s2_after_rgb)
    save_grayscale(event_out_dir / "s1_before.png", s1_before)
    save_grayscale(event_out_dir / "s1_after.png", s1_after)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].imshow(np.clip(s2_before_rgb, 0.0, 1.0))
    axes[0, 0].set_title("S2 BEFORE RGB")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(np.clip(s2_after_rgb, 0.0, 1.0))
    axes[0, 1].set_title("S2 AFTER RGB")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(normalize_display(s1_before), cmap="gray")
    axes[1, 0].set_title("S1 BEFORE")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(normalize_display(s1_after), cmap="gray")
    axes[1, 1].set_title("S1 AFTER")
    axes[1, 1].axis("off")

    fig.tight_layout()
    fig.savefig(event_out_dir / "before_after_comparison.png", dpi=200)
    plt.close(fig)

    print("Creating visualization only...")
    print("Created:")
    for name in [
        "s2_before_rgb.png",
        "s2_after_rgb.png",
        "s1_before.png",
        "s1_after.png",
        "before_after_comparison.png",
    ]:
        print(f"  {event_out_dir / name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Run inference on a single event from Canada HDF5 using CA-LMoETransUNet.

Usage:
  python3 predict_ca_lmoe.py --model best_model.pth --event CA_2017_BC_1157 [--h5 PATH] [--out-dir results/]
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from train_ca_lmoe import CALMoETransUNet
from unified_dataset import load_event_from_h5


def get_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def predict_full_image(model, img, device, patch_size=256, batch_size=8, overlap=0):
    """Run model on full image using patch sliding window."""
    C, H, W = img.shape
    stride = patch_size - overlap if overlap > 0 else patch_size

    # Calculate padded size
    pad_h = (patch_size - H % stride) % patch_size
    pad_w = (patch_size - W % stride) % patch_size

    if pad_h > 0 or pad_w > 0:
        img_padded = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    else:
        img_padded = img

    _, pad_H, pad_W = img_padded.shape

    prob_accum = np.zeros((pad_H, pad_W), dtype=np.float32)
    count_accum = np.zeros((pad_H, pad_W), dtype=np.float32)

    patches = []
    coords = []

    y_steps = list(range(0, pad_H - patch_size + 1, stride))
    if y_steps[-1] + patch_size < pad_H:
        y_steps.append(pad_H - patch_size)

    x_steps = list(range(0, pad_W - patch_size + 1, stride))
    if x_steps[-1] + patch_size < pad_W:
        x_steps.append(pad_W - patch_size)

    for y in y_steps:
        for x in x_steps:
            patch = img_padded[:, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            coords.append((y, x))

    for i in range(0, len(patches), batch_size):
        batch = np.stack(patches[i:i+batch_size])
        batch_tensor = torch.from_numpy(batch).to(device)
        batch_tensor = torch.nan_to_num(batch_tensor, nan=0.0, posinf=1.0, neginf=0.0)

        with torch.no_grad():
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        for j, (y, x) in enumerate(coords[i:i+batch_size]):
            prob_accum[y:y+patch_size, x:x+patch_size] += probs[j]
            count_accum[y:y+patch_size, x:x+patch_size] += 1.0

    count_accum = np.maximum(count_accum, 1.0)
    final_prob = prob_accum / count_accum
    return final_prob[:H, :W]


def main():
    parser = argparse.ArgumentParser(description="Predict burned area for HDF5 event")
    parser.add_argument("--h5",          default="candata/wildfire-s1s2-alos-dataset-canada-uint16.h5", help="HDF5 file path")
    parser.add_argument("--model",       required=True, help="Path to checkpoint best_model.pth")
    parser.add_argument("--event",       default="CA_2017_BC_1157", help="Event name to predict")
    parser.add_argument("--out-dir",     default="results", help="Output directory for PNG/NPY")
    parser.add_argument("--threshold",   type=float, default=0.5, help="Probability threshold for mask")
    parser.add_argument("--patch-size",  type=int,   default=256, help="Tiling patch size")
    parser.add_argument("--overlap",     type=int,   default=0,   help="Tiling overlap pixels")
    parser.add_argument("--batch-size",  type=int,   default=8,   help="Inference batch size")
    parser.add_argument("--device",      type=str,   default=None, help="Device (cuda, mps, cpu)")
    parser.add_argument("--embed-dim",   type=int,   default=128)
    parser.add_argument("--num-experts", type=int,   default=3)
    parser.add_argument("--depth",       type=int,   default=1)
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Using device fallback: {device}")

    model_path = Path(args.model)
    if not model_path.exists():
        if (Path("candata") / model_path).exists():
            model_path = Path("candata") / model_path
        elif Path("best_model.pth").exists():
            model_path = Path("best_model.pth")
        else:
            raise FileNotFoundError(f"Model file not found: {args.model}")

    print(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')

    embed_dim   = checkpoint.get('args', {}).get('embed_dim', args.embed_dim)
    num_experts = checkpoint.get('args', {}).get('num_experts', args.num_experts)
    depth       = checkpoint.get('args', {}).get('depth', args.depth)

    model = CALMoETransUNet(
        in_channels=12,
        out_channels=2,
        embed_dim=embed_dim,
        num_experts=num_experts,
        transformer_depth=depth
    )

    state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
    # Remove module. prefix if saved with DataParallel
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Model loaded successfully. Checkpoint val_dice: {checkpoint.get('val_dice', 0.0):.4f}")

    h5_path = Path(args.h5)
    if not h5_path.exists() and (Path("candata") / h5_path).exists():
        h5_path = Path("candata") / h5_file if 'h5_file' in locals() else Path("candata") / h5_path

    print(f"Loading event {args.event} from {h5_path} ...")
    img, mask_gt = load_event_from_h5(h5_path, args.event)

    # Check input channel count matching model input channels
    if img.shape[0] != model.enc1.conv1.in_channels:
        raise ValueError(
            f"Input channel count ({img.shape[0]}) does not match model expected in_channels ({model.enc1.conv1.in_channels})"
        )

    print("Running sliding window prediction ...")
    pred_prob = predict_full_image(
        model, img, device,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        overlap=args.overlap
    )

    pred_binary = (pred_prob > args.threshold).astype(np.uint8)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    event_basename = args.event.split('/')[-1]
    np.save(out_dir / f"{event_basename}_prob.npy", pred_prob)

    # Visualization
    r = img[6] # S2_pre B4 / Red
    g = img[7] # S2_pre B8 / NIR
    b = img[8] # S2_pre B12 / SWIR
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb * 3.0, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(rgb)
    axes[0].set_title(f"S2 RGB Preview\n{event_basename}")
    axes[0].axis('off')

    axes[1].imshow(mask_gt, cmap='gray')
    axes[1].set_title("Ground Truth (Poly)")
    axes[1].axis('off')

    axes[2].imshow(pred_prob, cmap='inferno')
    axes[2].set_title(f"Predicted Probability\n(Threshold={args.threshold})")
    axes[2].axis('off')

    axes[3].imshow(rgb)
    axes[3].imshow(np.where(pred_binary == 1, 1.0, np.nan), cmap='Reds', alpha=0.6, vmin=0, vmax=1)
    axes[3].set_title("Overlay (Prediction in Red)")
    axes[3].axis('off')

    plt.tight_layout()
    out_png = out_dir / f"prediction_{event_basename}.png"
    plt.savefig(out_png, dpi=150)
    plt.close()

    print(f"Prediction complete. Output saved to:\n  - {out_dir / f'{event_basename}_prob.npy'}\n  - {out_png}")


if __name__ == "__main__":
    main()

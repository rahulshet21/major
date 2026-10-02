#!/usr/bin/env python3
"""
Folder-based and HDF5-based Burned-Area Prediction Pipeline.

Supports:
  1. Input Folder containing S1 & S2 pre/post GeoTIFFs or .npy files
  2. Alternative HDF5 file input (--h5 <file> --event <name>)
  3. Structure inspection (--inspect)
  4. Device fallback: cuda -> mps -> cpu
  5. Georeferenced GeoTIFF mask output, probability .npy, and 4-panel visual PNG

Usage:
  python3 predict_folder.py --input_dir test_event_folder --weights best_model.pth --out_dir results/
  python3 predict_folder.py --h5 candata/wildfire-s1s2-alos-dataset-canada-uint16.h5 --event CA_2017_BC_1157 --weights best_model.pth --inspect
"""

import argparse
import os
import sys
import re
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

# Try importing rasterio for GeoTIFF metadata handling
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

# Local imports
try:
    from candata.train_ca_lmoe import CALMoETransUNet
    from candata.unified_dataset import load_event_from_h5, clean_and_normalize, resolve_existing_path
except ImportError:
    from train_ca_lmoe import CALMoETransUNet
    from unified_dataset import load_event_from_h5, clean_and_normalize, resolve_existing_path


def resolve_output_dir(out_dir_str):
    """Ensures outputs are always saved relative to project root /Users/shet/fire/results."""
    p = Path(out_dir_str)
    if not p.is_absolute():
        cwd = Path.cwd()
        if cwd.name == "candata":
            return (cwd.parent / p).resolve()
    return p.resolve()


def get_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_image_file(filepath):
    """
    Reads a raster image from GeoTIFF or .npy file.
    Returns: (numpy_array_float32, profile_dict_or_None)
    Array shape is always [C, H, W].
    """
    path = Path(filepath)
    profile = None

    if path.suffix.lower() in ['.tif', '.tiff']:
        if not HAS_RASTERIO:
            raise RuntimeError("rasterio is required to read GeoTIFF files. Please install via: pip install rasterio")
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)
            profile = src.profile.copy()
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            return arr, profile
    elif path.suffix.lower() == '.npy':
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        return arr, None
    else:
        raise ValueError(f"Unsupported file format for raster reading: {path}")


def extract_date_or_name(file_path):
    """Extracts date YYYY-MM-DD from filename if present, otherwise returns filename string."""
    name = file_path.name.lower()
    m = re.search(r'(\d{4}[-_]\d{2}[-_]\d{2})', name)
    if m:
        return m.group(1)
    return name


def find_folder_rasters(input_dir):
    """
    Auto-discovers the 4 required S1/S2 pre/post files and optional mask in input_dir.
    Matches filenames case-insensitively, supporting s1, s2, sentinel1, sentinel2,
    and sorting by pre/post keywords or chronological date in filename.
    """
    input_path = resolve_existing_path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir} (resolved as {input_path})")

    tifs  = sorted(list(input_path.glob("*.tif")) + list(input_path.glob("*.tiff")))
    npys  = sorted(list(input_path.glob("*.npy")))
    files = [f for f in (tifs + npys) if not f.name.endswith("_coverage.png") and not f.name.endswith(".png")]

    if not files:
        raise FileNotFoundError(f"No .tif, .tiff, or .npy raster files found in {input_dir}")

    s1_files = []
    s2_files = []
    mask_file = None

    for f in files:
        name = f.name.lower()
        if 'mask' in name and mask_file is None:
            mask_file = f
        elif any(k in name for k in ['s1', 'sentinel1', 'sentinel-1', 'sentinel_1']):
            s1_files.append(f)
        elif any(k in name for k in ['s2', 'sentinel2', 'sentinel-2', 'sentinel_2']):
            s2_files.append(f)

    s1_pre, s1_post = None, None
    s2_pre, s2_post = None, None

    for f in s1_files:
        name = f.name.lower()
        if 'pre' in name or 'before' in name:
            s1_pre = f
        elif 'post' in name or 'after' in name:
            s1_post = f

    if (not s1_pre or not s1_post) and len(s1_files) >= 2:
        s1_sorted = sorted(s1_files, key=extract_date_or_name)
        s1_pre = s1_sorted[0]
        s1_post = s1_sorted[-1]

    for f in s2_files:
        name = f.name.lower()
        if 'pre' in name or 'before' in name:
            s2_pre = f
        elif 'post' in name or 'after' in name:
            s2_post = f

    if (not s2_pre or not s2_post) and len(s2_files) >= 2:
        s2_sorted = sorted(s2_files, key=extract_date_or_name)
        s2_pre = s2_sorted[0]
        s2_post = s2_sorted[-1]

    missing = []
    if not s1_pre: missing.append("Sentinel-1 Pre")
    if not s1_post: missing.append("Sentinel-1 Post")
    if not s2_pre: missing.append("Sentinel-2 Pre")
    if not s2_post: missing.append("Sentinel-2 Post")

    if missing:
        found_names = [f.name for f in files]
        raise ValueError(
            f"Could not auto-discover required raster files in '{input_dir}'.\n"
            f"Missing: {', '.join(missing)}\n"
            f"Found files in directory: {found_names}"
        )

    return {
        "s1_pre": s1_pre,
        "s1_post": s1_post,
        "s2_pre": s2_pre,
        "s2_post": s2_post,
        "mask": mask_file
    }


def prepare_folder_input(input_dir):
    """
    Auto-discovers and stacks the 4 raster files into [12, H, W] float32 tensor
    with training channel order: [s1_pre(3), s1_post(3), s2_pre(3), s2_post(3)].
    """
    found = find_folder_rasters(input_dir)

    arr_s1_pre, prof = read_image_file(found["s1_pre"])
    arr_s1_post, _  = read_image_file(found["s1_post"])
    arr_s2_pre, _   = read_image_file(found["s2_pre"])
    arr_s2_post, _  = read_image_file(found["s2_post"])

    mask_gt = None
    if found["mask"]:
        arr_mask, _ = read_image_file(found["mask"])
        mask_gt = (arr_mask[0] > 0).astype(np.int64)

    def extract_3ch_s1(arr):
        if arr.shape[0] >= 3:
            return arr[:3]
        elif arr.shape[0] == 2:
            nd = (arr[0] - arr[1]) / (arr[0] + arr[1] + 1e-6)
            return np.concatenate([nd[np.newaxis, ...], arr], axis=0)
        elif arr.shape[0] == 1:
            return np.repeat(arr, 3, axis=0)
        return arr

    def extract_3ch_s2(arr):
        if arr.shape[0] >= 12:
            return arr[[3, 7, 11]]
        elif arr.shape[0] >= 3:
            return arr[:3]
        elif arr.shape[0] == 1:
            return np.repeat(arr, 3, axis=0)
        return arr

    s1_pre  = extract_3ch_s1(arr_s1_pre)
    s1_post = extract_3ch_s1(arr_s1_post)
    s2_pre  = extract_3ch_s2(arr_s2_pre)
    s2_post = extract_3ch_s2(arr_s2_post)

    stacked_raw = np.concatenate([s1_pre, s1_post, s2_pre, s2_post], axis=0)
    img = clean_and_normalize(stacked_raw, is_uint16=False)

    return img, mask_gt, prof, found


def predict_tiled(model, img, device, patch_size=256, batch_size=8, overlap=0):
    """Tiles the image into patch_size patches with optional overlap, predicts, and stitches."""
    C, H, W = img.shape
    stride = patch_size - overlap if overlap > 0 else patch_size

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


def save_geotiff(out_path, mask_array, reference_profile):
    """Saves binary mask array as a georeferenced GeoTIFF if reference profile exists."""
    if not HAS_RASTERIO or reference_profile is None:
        return False

    prof = reference_profile.copy()
    prof.update(
        dtype=rasterio.uint8,
        count=1,
        nodata=0,
        compress='lzw'
    )
    with rasterio.open(out_path, 'w', **prof) as dst:
        dst.write(mask_array.astype(np.uint8), 1)
    return True


def save_visualization(out_png_path, img_12ch, pred_prob, mask_gt, threshold, event_name="Burned Area"):
    """Saves a 4-panel visual PNG (RGB preview, ground truth, probability heatmap, red overlay)."""
    r = img_12ch[6]
    g = img_12ch[7]
    b = img_12ch[8]
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb * 3.0, 0, 1)

    pred_binary = (pred_prob > threshold).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(rgb)
    axes[0].set_title(f"S2 Pre-Fire RGB\n{event_name}")
    axes[0].axis('off')

    if mask_gt is not None:
        axes[1].imshow(mask_gt, cmap='gray')
        axes[1].set_title("Ground Truth Mask")
    else:
        axes[1].text(0.5, 0.5, "No Ground Truth\nMask Provided", ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title("Ground Truth (N/A)")
    axes[1].axis('off')

    axes[2].imshow(pred_prob, cmap='inferno')
    axes[2].set_title(f"Predicted Probability\n(Thresh={threshold})")
    axes[2].axis('off')

    axes[3].imshow(rgb)
    axes[3].imshow(np.where(pred_binary == 1, 1.0, np.nan), cmap='Reds', alpha=0.6, vmin=0, vmax=1)
    axes[3].set_title("Overlay (Prediction in Red)")
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(out_png_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Predict burned area from folder of rasters or HDF5 event")
    parser.add_argument("--input_dir",   default=None, help="Input directory containing S1/S2 TIFF or .npy files")
    parser.add_argument("--h5",          default="wildfire-s1s2-alos-dataset-canada-uint16.h5", help="Alternative input: Path to Canada HDF5 file")
    parser.add_argument("--event",       default=None, help="Event name when using --h5 (e.g. CA_2017_BC_1157)")
    parser.add_argument("--weights",     default="best_model.pth", help="Path to model weights (.pth)")
    parser.add_argument("--out_dir",     default="results", help="Output directory for predictions")
    parser.add_argument("--threshold",   type=float, default=0.5, help="Probability threshold for mask")
    parser.add_argument("--patch",       type=int,   default=256, help="Tiling patch size")
    parser.add_argument("--overlap",     type=int,   default=0,   help="Overlap pixels for sliding window")
    parser.add_argument("--device",      type=str,   default=None, help="Device (cuda, mps, cpu)")
    parser.add_argument("--inspect",     action="store_true", help="Inspect input files/structure and exit")
    args = parser.parse_args()

    if args.input_dir is None and args.event is None and not args.inspect:
        parser.error("Must specify either --input_dir <folder> OR --h5 <file> --event <name>")

    device = get_device(args.device)

    # 1. Prepare Input Data
    profile_geotiff = None
    if args.event or (args.input_dir is None and args.h5):
        event_name_str = args.event if args.event else "CA_2017_BC_1157"
        h5_path = resolve_existing_path(args.h5)
        print(f"Loading event {event_name_str} from HDF5: {h5_path}")
        img, mask_gt = load_event_from_h5(h5_path, event_name_str)
        event_name = event_name_str.split('/')[-1]
    else:
        print(f"Auto-discovering raster files in directory: {args.input_dir}")
        img, mask_gt, profile_geotiff, found_files = prepare_folder_input(args.input_dir)
        event_name = Path(args.input_dir).name
        if args.inspect:
            print("\n=== INPUT FILE INSPECTION ===")
            for k, v in found_files.items():
                print(f"  {k:10s}: {v}")
            print(f"  Stacked Tensor Shape : {img.shape} [Channels, Height, Width]")
            print(f"  Data Type            : {img.dtype}")
            print(f"  Value Range          : min={img.min():.4f}, max={img.max():.4f}")
            if profile_geotiff:
                print(f"  CRS                  : {profile_geotiff.get('crs')}")
            else:
                print("  CRS                  : None (No GeoTIFF CRS found)")
            print("=============================\n")

    if args.inspect and (args.event or args.input_dir is None):
        print("\n=== HDF5 EVENT INSPECTION ===")
        print(f"  Event Name  : {event_name}")
        print(f"  Image Shape : {img.shape} [Channels, Height, Width]")
        print(f"  Data Type   : {img.dtype}")
        print(f"  Value Range : min={img.min():.4f}, max={img.max():.4f}")
        if mask_gt is not None:
            print(f"  Mask Shape  : {mask_gt.shape}, Burned Pixels: {(mask_gt == 1).sum()}")
        print("=============================\n")

    if args.inspect:
        print("--inspect flag set. Exiting without running prediction.")
        sys.exit(0)

    # 2. Load Checkpoint & Real Model Class
    weights_path = resolve_existing_path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {args.weights} (resolved as {weights_path})")

    print(f"Loading checkpoint: {weights_path} (Using device: {device})")
    checkpoint = torch.load(weights_path, map_location='cpu')

    ckpt_args = checkpoint.get('args', {})
    embed_dim   = ckpt_args.get('embed_dim', 128)
    num_experts = ckpt_args.get('num_experts', 3)
    depth       = ckpt_args.get('depth', 1)

    model = CALMoETransUNet(
        in_channels=12,
        out_channels=2,
        embed_dim=embed_dim,
        num_experts=num_experts,
        transformer_depth=depth
    )

    state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. Channel Count Assertion (Required Task 4 Check)
    expected_channels = model.enc1.conv1.in_channels
    if img.shape[0] != expected_channels:
        raise ValueError(
            f"STRICT CHANNEL MISMATCH ERROR: Stacked input tensor has {img.shape[0]} channels, "
            f"but model architecture expects {expected_channels} channels. "
            f"Expected channels: [s1_pre(3), s1_post(3), s2_pre(3), s2_post(3)]."
        )

    print("Running tiled prediction with sliding window ...")
    pred_prob = predict_tiled(
        model, img, device,
        patch_size=args.patch,
        overlap=args.overlap
    )

    pred_binary = (pred_prob > args.threshold).astype(np.uint8)

    out_dir = resolve_output_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4. Save Outputs
    prob_npy_path = out_dir / f"{event_name}_prob.npy"
    np.save(prob_npy_path, pred_prob)
    print(f"Saved probability array -> {prob_npy_path}")

    if profile_geotiff is not None:
        geotiff_path = out_dir / f"{event_name}_mask.tif"
        tif_saved = save_geotiff(geotiff_path, pred_binary, profile_geotiff)
        if tif_saved:
            print(f"Saved georeferenced GeoTIFF mask -> {geotiff_path}")

    png_path = out_dir / f"{event_name}_prediction.png"
    save_visualization(png_path, img, pred_prob, mask_gt, args.threshold, event_name=event_name)
    print(f"Saved visualization PNG -> {png_path}")

    print(f"\nPrediction completed successfully!")
    print(f"Outputs written to: {out_dir}/")


if __name__ == "__main__":
    main()

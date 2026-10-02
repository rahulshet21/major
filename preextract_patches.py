#!/usr/bin/env python3
"""
Pre-extract patches from Canada HDF5 dataset into numpy arrays for fast training.
Run this once, then train_ca_lmoe.py will use the cached patches.

Usage:
  python3 preextract_patches.py [--h5 PATH] [--patches 16] [--patch-size 256] [--out patches_ca.npz]
"""

import argparse
import random
import time
from pathlib import Path

import h5py
import numpy as np
from unified_dataset import clean_and_normalize


def get_device(requested_device=None):
    if requested_device:
        return requested_device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Pre-extract patches from wildfire HDF5 dataset")
    parser.add_argument("--h5",          default="candata/wildfire-s1s2-alos-dataset-canada-uint16.h5", help="Path to HDF5 file")
    parser.add_argument("--groups-txt",  default="candata/wildfire-s1s2-alos-dataset-canada-groups.txt", help="Path to groups txt file")
    parser.add_argument("--patches",     type=int, default=16, help="Patches per event")
    parser.add_argument("--patch-size",  type=int, default=256, help="Patch height/width")
    parser.add_argument("--out",         default="patches_ca.npz", help="Output .npz file path")
    parser.add_argument("--out-images",  default="patches_images.npy", help="Output .npy file path for images")
    parser.add_argument("--out-masks",   default="patches_masks.npy", help="Output .npy file path for masks")
    parser.add_argument("--event",       default=None, help="Single event filter (optional)")
    parser.add_argument("--seed",        type=int, default=42, help="Random seed")
    parser.add_argument("--device",      type=str, default=None, help="Device (cuda, mps, cpu)")
    args = parser.parse_args()

    device_str = get_device(args.device)
    print(f"Using device fallback: {device_str}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    P = args.patch_size

    h5_path = Path(args.h5)
    if not h5_path.exists():
        # Fallback search if path is relative to candata
        if (Path("candata") / h5_path).exists():
            h5_path = Path("candata") / h5_path
        else:
            raise FileNotFoundError(f"HDF5 file not found: {args.h5}")

    groups_txt_path = Path(args.groups_txt)
    all_events = []
    if groups_txt_path.exists():
        with open(groups_txt_path) as f:
            for line in f:
                line = line.strip().strip("'()\r\n ,")
                if line.startswith('/20') and line.count('/') == 2:
                    all_events.append(line.strip('/'))
    else:
        with h5py.File(h5_path, 'r') as f:
            for y in f.keys():
                for ev in f[y].keys():
                    all_events.append(f"{y}/{ev}")

    if args.event:
        all_events = [e for e in all_events if args.event in e]

    print(f"Events to process: {len(all_events)}")

    images_list = []
    masks_list  = []
    t0 = time.time()

    with h5py.File(h5_path, 'r') as f:
        for idx, group in enumerate(all_events):
            try:
                g = f[group]
                _, H, W = g['mask'].shape
                if H < P or W < P:
                    continue

                for _ in range(args.patches):
                    y0 = random.randint(0, H - P)
                    x0 = random.randint(0, W - P)

                    s1_pre  = g['s1'][0, :, y0:y0+P, x0:x0+P]
                    s1_post = g['s1'][1, :, y0:y0+P, x0:x0+P]
                    s2_pre  = g['s2'][0, :, y0:y0+P, x0:x0+P]
                    s2_post = g['s2'][1, :, y0:y0+P, x0:x0+P]
                    mask    = g['mask'][0, y0:y0+P, x0:x0+P].astype(np.int64)

                    raw_img = np.concatenate([s1_pre, s1_post, s2_pre, s2_post], axis=0)
                    img = clean_and_normalize(raw_img, is_uint16=True)
                    images_list.append(img)
                    masks_list.append((mask > 0).astype(np.int8))

                elapsed = time.time() - t0
                speed   = (idx + 1) / elapsed if elapsed > 0 else 0
                eta     = (len(all_events) - idx - 1) / speed if speed > 0 else 0
                print(f"  [{idx+1:3d}/{len(all_events)}] {group:30s}  "
                      f"patches={len(images_list)}  ETA={eta:.0f}s", end='\r', flush=True)

            except Exception as e:
                print(f"\n  Skipping {group}: {e}")

    print(f"\nExtracted {len(images_list)} patches in {time.time()-t0:.1f}s")

    if not images_list:
        print("No patches extracted.")
        return

    images = np.stack(images_list, axis=0)   # (N, 12, P, P)
    masks  = np.stack(masks_list,  axis=0)   # (N, P, P)
    print(f"Images: {images.shape}  Masks: {masks.shape}")

    np.savez_compressed(args.out, images=images, masks=masks)
    print(f"Saved {args.out}")

    np.save(args.out_images, images)
    np.save(args.out_masks, masks)
    print(f"Saved {args.out_images} and {args.out_masks}")


if __name__ == "__main__":
    main()

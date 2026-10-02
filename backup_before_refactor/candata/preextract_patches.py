#!/usr/bin/env python3
"""
Pre-extract patches from the Canada HDF5 into a single .npz file for fast training.
Run this once, then train_ca_lmoe.py will use the cached patches.

Usage:
  python3 preextract_patches.py [--patches 16] [--patch-size 256] [--out patches_ca.npz]
"""
import argparse
import random
import time
from pathlib import Path

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5",          default="wildfire-s1s2-alos-dataset-canada-uint16.h5")
    parser.add_argument("--groups-txt",  default="wildfire-s1s2-alos-dataset-canada-groups.txt")
    parser.add_argument("--patches",     type=int, default=16, help="Patches per event")
    parser.add_argument("--patch-size",  type=int, default=256)
    parser.add_argument("--out",         default="patches_ca.npz")
    parser.add_argument("--event",       default=None, help="Single event (optional)")
    args = parser.parse_args()

    random.seed(42); np.random.seed(42)
    P = args.patch_size

    # Read event list
    all_events = []
    with open(args.groups_txt) as f:
        for line in f:
            line = line.strip().strip("'()\r\n ,")
            if line.startswith('/20') and line.count('/') == 2:
                all_events.append(line.strip('/'))  # '2017/CA_2017_BC_1157'

    if args.event:
        all_events = [e for e in all_events if args.event in e]

    print(f"Events to process: {len(all_events)}")

    images_list = []
    masks_list  = []
    t0 = time.time()

    with h5py.File(args.h5, 'r') as f:
        for idx, group in enumerate(all_events):
            try:
                g = f[group]
                _, H, W = g['mask'].shape
                if H < P or W < P:
                    continue

                for _ in range(args.patches):
                    y0 = random.randint(0, H - P)
                    x0 = random.randint(0, W - P)

                    s1_pre  = g['s1'][0, :, y0:y0+P, x0:x0+P].astype(np.float32)
                    s1_post = g['s1'][1, :, y0:y0+P, x0:x0+P].astype(np.float32)
                    s2_pre  = g['s2'][0, :, y0:y0+P, x0:x0+P].astype(np.float32)
                    s2_post = g['s2'][1, :, y0:y0+P, x0:x0+P].astype(np.float32)
                    mask    = g['mask'][0, y0:y0+P, x0:x0+P].astype(np.int64)

                    img = np.concatenate([s1_pre, s1_post, s2_pre, s2_post], axis=0) / 65535.0
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

    images = np.stack(images_list, axis=0)   # (N, 12, P, P)
    masks  = np.stack(masks_list,  axis=0)   # (N, P, P)
    print(f"Images: {images.shape}  Masks: {masks.shape}")
    print(f"Saving to {args.out} ...")
    np.savez_compressed(args.out, images=images, masks=masks)
    size_mb = Path(args.out).stat().st_size / 1e6
    print(f"Done! Saved {args.out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

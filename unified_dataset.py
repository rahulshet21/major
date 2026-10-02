"""
Unified Dataset Loader and Data Adapter for Wildfire Burned-Area Mapping.

Provides a unified interface for loading both:
  1. Canada HDF5 dataset (wildfire-s1s2-alos-dataset-canada-uint16.h5)
  2. Legacy European/Wildfire 30-channel .npy dataset (from wildfire/ processed/)
  3. Pre-extracted memory-mapped .npy patch arrays (patches_images.npy, patches_masks.npy)

All outputs are guaranteed to be identical:
  - Image Tensor: float32 [12, H, W] normalized to [0.0, 1.0]
  - Fixed Channel Order:
      Channels 0..2  : Sentinel-1 Pre-fire  (3 bands: ND, VH, VV)
      Channels 3..5  : Sentinel-1 Post-fire (3 bands: ND, VH, VV)
      Channels 6..8  : Sentinel-2 Pre-fire  (3 bands: B4, B8, B12)
      Channels 9..11 : Sentinel-2 Post-fire (3 bands: B4, B8, B12)
  - Target Mask: int64 [H, W] binary mask (0 = unburned, 1 = burned)
"""

import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py

CHANNEL_ORDER = [
    "s1_pre_0", "s1_pre_1", "s1_pre_2",
    "s1_post_0", "s1_post_1", "s1_post_2",
    "s2_pre_0", "s2_pre_1", "s2_pre_2",
    "s2_post_0", "s2_post_1", "s2_post_2"
]


def resolve_existing_path(path_str):
    """Smart path resolver that checks CWD, subfolders, and parent directories."""
    if not path_str:
        return Path(path_str)
    p = Path(path_str)
    if p.exists():
        return p.resolve()
    if p.parts and p.parts[0] == "candata":
        stripped = Path(*p.parts[1:])
        if stripped.exists():
            return stripped.resolve()
    if (Path("candata") / p).exists():
        return (Path("candata") / p).resolve()
    if (Path("..") / p).exists():
        return (Path("..") / p).resolve()
    if p.parts and p.parts[0] == "candata" and (Path("..") / Path(*p.parts[1:])).exists():
        return (Path("..") / Path(*p.parts[1:])).resolve()
    return p


def clean_and_normalize(img_arr, is_uint16=True):
    """
    Robust sensor data cleaner & normalizer.
    Evaluates each channel independently to handle raw uint16 (0..65535),
    float reflectance (0..1/2), dB radar values, and removes Inf/NaN outliers.
    """
    arr = img_arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    out = np.zeros_like(arr, dtype=np.float32)
    for i in range(arr.shape[0]):
        band = arr[i].copy()
        # Clean unphysical extreme outliers (e.g. Inf converted to float32 max)
        band[band > 65535.0] = 0.0
        band[band < -100.0] = 0.0

        valid = (band > 0) & np.isfinite(band)
        if not np.any(valid):
            continue

        vmax = float(band[valid].max())
        if vmax > 10.0:
            if vmax <= 10000.0:
                out[i] = np.clip(band / 10000.0, 0.0, 1.0)
            else:
                out[i] = np.clip(band / 65535.0, 0.0, 1.0)
        else:
            # Float32 values (0.0 to 2.0 reflectance / radar ratio): use 2-98% percentile stretch
            low = float(np.percentile(band[valid], 2))
            high = float(np.percentile(band[valid], 98))
            if high > low:
                out[i] = np.clip((band - low) / (high - low), 0.0, 1.0)
            else:
                out[i] = np.clip(band, 0.0, 1.0)
    return out


def adapt_legacy_30ch_to_12ch(img_30ch):
    """Adapter for legacy 30-channel dataset (wildfire/processed/)."""
    if img_30ch.shape[0] != 30:
        raise ValueError(f"Expected 30 channels for legacy adapter, got {img_30ch.shape[0]}")
    
    s1_pre  = img_30ch[12:15]
    s1_post = img_30ch[27:30]
    s2_pre  = img_30ch[[3, 7, 11]]
    s2_post = img_30ch[[18, 22, 26]]
    
    img_12ch = np.concatenate([s1_pre, s1_post, s2_pre, s2_post], axis=0)
    return clean_and_normalize(img_12ch, is_uint16=False)


def find_hdf5_event_group(h5_file, event_name):
    """Locate event group in Canada HDF5 regardless of year prefix format."""
    event_clean = event_name.strip('/')
    if '/' in event_clean:
        if event_clean in h5_file:
            return h5_file[event_clean]
    
    for year in h5_file.keys():
        if event_clean in h5_file[year]:
            return h5_file[year][event_clean]
        full_path = f"{year}/{event_clean}"
        if full_path in h5_file:
            return h5_file[full_path]
            
    raise KeyError(f"Event '{event_name}' not found in HDF5 file.")


def load_event_from_h5(h5_path, event_name):
    """
    Loads full 12-channel image and mask for a specific event from Canada HDF5.
    """
    resolved_h5 = resolve_existing_path(h5_path)
    if not resolved_h5.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path} (resolved as {resolved_h5})")

    with h5py.File(resolved_h5, 'r') as f:
        g = find_hdf5_event_group(f, event_name)
        s1_pre  = g['s1'][0, :, :, :]
        s1_post = g['s1'][1, :, :, :]
        s2_pre  = g['s2'][0, :, :, :]
        s2_post = g['s2'][1, :, :, :]
        mask    = g['mask'][0, :, :].astype(np.int64)

        raw_img = np.concatenate([s1_pre, s1_post, s2_pre, s2_post], axis=0)
        img = clean_and_normalize(raw_img, is_uint16=True)
        mask = (mask > 0).astype(np.int64)
        return img, mask


class CanadaHDF5Dataset(Dataset):
    """Patch dataset loading directly from Canada HDF5 file."""
    def __init__(self, h5_path, patch_size=256, patches_per_event=16, events=None, augment=True):
        self.h5_path = str(resolve_existing_path(h5_path))
        self.patch_size = patch_size
        self.augment = augment
        self.samples = []

        groups_txt = Path(self.h5_path).with_name(
            Path(self.h5_path).stem.replace('-uint16', '') + '-groups.txt'
        )
        all_events = []
        if groups_txt.exists():
            with open(groups_txt) as f:
                for line in f:
                    line = line.strip().strip("'()\r\n ,")
                    if line.startswith('/20') and line.count('/') == 2:
                        all_events.append(line.strip('/'))
        else:
            with h5py.File(self.h5_path, 'r') as f:
                for y in f.keys():
                    for ev in f[y].keys():
                        all_events.append(f"{y}/{ev}")

        if events:
            all_events = [e for e in all_events if any(ev in e for ev in events)]

        with h5py.File(self.h5_path, 'r') as f:
            for group in all_events:
                try:
                    g = f[group]
                    _, H, W = g['mask'].shape
                    if H < patch_size or W < patch_size:
                        continue
                    for _ in range(patches_per_event):
                        y0 = random.randint(0, H - patch_size)
                        x0 = random.randint(0, W - patch_size)
                        self.samples.append((group, y0, x0))
                except Exception:
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        group, y0, x0 = self.samples[idx]
        P = self.patch_size
        with h5py.File(self.h5_path, 'r') as f:
            g = f[group]
            s1_pre  = g['s1'][0, :, y0:y0+P, x0:x0+P]
            s1_post = g['s1'][1, :, y0:y0+P, x0:x0+P]
            s2_pre  = g['s2'][0, :, y0:y0+P, x0:x0+P]
            s2_post = g['s2'][1, :, y0:y0+P, x0:x0+P]
            mask    = g['mask'][0, y0:y0+P, x0:x0+P].astype(np.int64)

        raw_img = np.concatenate([s1_pre, s1_post, s2_pre, s2_post], axis=0)
        img = clean_and_normalize(raw_img, is_uint16=True)

        img_t  = torch.from_numpy(img)
        mask_t = torch.from_numpy((mask > 0).astype(np.int64))

        if self.augment:
            if random.random() > 0.5:
                img_t  = torch.flip(img_t,  dims=[-1])
                mask_t = torch.flip(mask_t, dims=[-1])
            if random.random() > 0.5:
                img_t  = torch.flip(img_t,  dims=[-2])
                mask_t = torch.flip(mask_t, dims=[-2])

        return img_t, mask_t


class LegacyNpyDataset(Dataset):
    """Patch dataset loading from legacy processed .npy files (30-channel or 12-channel)."""
    def __init__(self, processed_dir, patch_size=256, patches_per_event=16, event_list=None, augment=True):
        self.processed_dir = resolve_existing_path(processed_dir)
        self.patch_size = patch_size
        self.augment = augment
        self.samples = []

        if event_list:
            events = event_list
        else:
            events = [p.stem.replace('_image', '') for p in self.processed_dir.glob('*_image.npy')]

        for ev in events:
            img_p  = self.processed_dir / f"{ev}_image.npy"
            mask_p = self.processed_dir / f"{ev}_mask.npy"
            if not img_p.exists() or not mask_p.exists():
                continue
            img_arr = np.load(img_p)
            H, W = img_arr.shape[1], img_arr.shape[2]
            if H < patch_size or W < patch_size:
                continue
            for _ in range(patches_per_event):
                y0 = random.randint(0, H - patch_size)
                x0 = random.randint(0, W - patch_size)
                self.samples.append((ev, y0, x0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ev, y0, x0 = self.samples[idx]
        P = self.patch_size
        img_arr  = np.load(self.processed_dir / f"{ev}_image.npy")
        mask_arr = np.load(self.processed_dir / f"{ev}_mask.npy")

        img_patch  = img_arr[:, y0:y0+P, x0:x0+P]
        mask_patch = mask_arr[y0:y0+P, x0:x0+P]

        if img_patch.shape[0] == 30:
            img_patch = adapt_legacy_30ch_to_12ch(img_patch)
        else:
            img_patch = clean_and_normalize(img_patch, is_uint16=False)

        img_t  = torch.from_numpy(img_patch)
        mask_t = torch.from_numpy((mask_patch > 0).astype(np.int64))

        if self.augment:
            if random.random() > 0.5:
                img_t  = torch.flip(img_t,  dims=[-1])
                mask_t = torch.flip(mask_t, dims=[-1])
            if random.random() > 0.5:
                img_t  = torch.flip(img_t,  dims=[-2])
                mask_t = torch.flip(mask_t, dims=[-2])

        return img_t, mask_t


class MemmapDataset(Dataset):
    """Fast memory-mapped dataset from patches_images.npy and patches_masks.npy."""
    def __init__(self, images_npy, masks_npy, augment=True):
        images_path = resolve_existing_path(images_npy)
        masks_path  = resolve_existing_path(masks_npy)
        self.images  = np.load(images_path, mmap_mode='r')
        self.masks   = np.load(masks_path,  mmap_mode='r')
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img  = torch.from_numpy(self.images[idx].copy())
        mask = torch.from_numpy(self.masks[idx].astype(np.int64))
        
        img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
        img = img.clamp(0.0, 1.0)

        if self.augment:
            if random.random() > 0.5:
                img  = torch.flip(img,  dims=[-1])
                mask = torch.flip(mask, dims=[-1])
            if random.random() > 0.5:
                img  = torch.flip(img,  dims=[-2])
                mask = torch.flip(mask, dims=[-2])
        return img, mask

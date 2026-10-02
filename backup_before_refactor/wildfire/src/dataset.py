import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed"
SPLITS_DIR = PROJECT_ROOT / "splits"


def read_event_names(split_name):
    path = SPLITS_DIR / split_name
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_event_data(event_name, experiment):
    image_path = PROCESSED_DIR / f"{event_name}_image.npy"
    mask_path = PROCESSED_DIR / f"{event_name}_mask.npy"
    if not image_path.exists() or not mask_path.exists():
        raise FileNotFoundError(f"Event data missing: {event_name}")

    image = np.load(image_path).astype(np.float32)
    mask = np.load(mask_path).astype(np.uint8)

    if experiment == "fusion":
        data = image
    elif experiment == "s2":
        data = image[:12]
    elif experiment == "s1":
        data = image[12:15]
    else:
        raise ValueError(f"Unsupported experiment: {experiment}")

    return data, mask


def extract_patches(image, mask, patch_size=256, overlap=64, max_patches_per_event=None):
    if image.shape[1:] != mask.shape:
        raise ValueError(f"Image/mask spatial mismatch: {image.shape}, {mask.shape}")

    h, w = image.shape[1], image.shape[2]
    stride = patch_size - overlap

    starts_y = [0]
    while starts_y[-1] + patch_size < h:
        starts_y.append(starts_y[-1] + stride)
    if starts_y[-1] + patch_size < h:
        starts_y.append(h - patch_size)
    if starts_y[-1] != 0 and starts_y[-1] + patch_size > h:
        starts_y[-1] = max(0, h - patch_size)

    starts_x = [0]
    while starts_x[-1] + patch_size < w:
        starts_x.append(starts_x[-1] + stride)
    if starts_x[-1] + patch_size < w:
        starts_x.append(w - patch_size)
    if starts_x[-1] != 0 and starts_x[-1] + patch_size > w:
        starts_x[-1] = max(0, w - patch_size)

    patches = []
    for y0 in sorted(set(starts_y)):
        for x0 in sorted(set(starts_x)):
            y1 = min(y0 + patch_size, h)
            x1 = min(x0 + patch_size, w)
            img_patch = image[:, y0:y1, x0:x1]
            mask_patch = mask[y0:y1, x0:x1]
            if img_patch.shape[1] != patch_size or img_patch.shape[2] != patch_size:
                pad_h = patch_size - img_patch.shape[1]
                pad_w = patch_size - img_patch.shape[2]
                img_patch = np.pad(img_patch, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
                mask_patch = np.pad(mask_patch, ((0, pad_h), (0, pad_w)), mode="reflect")
            patches.append((img_patch.astype(np.float32), mask_patch.astype(np.uint8)))

    if max_patches_per_event is not None:
        if len(patches) > max_patches_per_event:
            rng = random.Random(42)
            patches = rng.sample(patches, max_patches_per_event)

    return patches


class EventPatchDataset(Dataset):
    def __init__(self, event_names, experiment, patch_size=256, overlap=64, max_patches_per_event=None):
        self.event_names = event_names
        self.experiment = experiment
        self.patch_size = patch_size
        self.overlap = overlap
        self.max_patches_per_event = max_patches_per_event
        self.patches = []

        for event_name in event_names:
            image, mask = load_event_data(event_name, experiment)
            patch_list = extract_patches(
                image,
                mask,
                patch_size=patch_size,
                overlap=overlap,
                max_patches_per_event=max_patches_per_event,
            )
            self.patches.extend([(event_name, img_patch, mask_patch) for img_patch, mask_patch in patch_list])

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        _, image_patch, mask_patch = self.patches[idx]
        image = torch.from_numpy(image_patch).float()
        mask = torch.from_numpy(mask_patch).float().unsqueeze(0)
        return image, mask


def iter_event_patches(event_names, experiment, patch_size=256, overlap=64, max_patches_per_event=None):
    for event_name in event_names:
        image, mask = load_event_data(event_name, experiment)
        for img_patch, mask_patch in extract_patches(
            image,
            mask,
            patch_size=patch_size,
            overlap=overlap,
            max_patches_per_event=max_patches_per_event,
        ):
            yield img_patch, mask_patch

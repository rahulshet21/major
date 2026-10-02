#!/usr/bin/env python3
"""
Train CA-LMoETransUNet on the Canada wildfire S1/S2 HDF5 dataset.

Dataset channels available per event:
  s1  : 3 bands (ND, VH, VV)  x 2 times (pre, post)  = 6 ch
  s2  : 3 bands (B4, B8, B12) x 2 times (pre, post)  = 6 ch
  alos: 3 bands (ND, HV, HH)  x 2 times (pre, post)  = 6 ch
  -------------------------------------------------------
  Total used: 12 ch  (s1 + s2)
  Model adapted to in_channels=12

Mask used: 'poly' band (NBAC burned area polygon).
Uses h5py directly for fast indexing and patch loading.

Usage:
  python3 train_ca_lmoe.py [--epochs 20] [--batch-size 4] [--patch-size 256]
                            [--lr 1e-4] [--event CA_2017_BC_1157]
"""

import argparse
import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import h5py

# ------------------------------------------------------------------
# 1.  Model  (adapted from CA_LMoETransUNet.ipynb)
# ------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout(dropout)
        self.short = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch)
        ) if in_ch != out_ch else nn.Sequential()

    def forward(self, x):
        s = self.short(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.bn2(self.conv2(x))
        return self.relu(x + s)


class MoEAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, num_experts=4, dropout=0.4):
        super().__init__()
        self.num_heads   = num_heads
        self.num_experts = num_experts
        self.qkv_proj    = nn.ModuleList([nn.Linear(dim, dim * 3) for _ in range(num_experts)])
        self.out_proj    = nn.ModuleList([nn.Linear(dim, dim)     for _ in range(num_experts)])
        self.gate        = nn.Linear(dim, num_experts)
        self.drop        = nn.Dropout(dropout)
        self.norm1       = nn.LayerNorm(dim)
        self.norm2       = nn.LayerNorm(dim)
        self.mlp         = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.ReLU(inplace=True), nn.Linear(mlp_dim, dim)
        )

    def forward(self, x):
        B, L, D = x.shape
        xn = self.norm1(x)
        gate_probs = F.softmax(self.gate(xn), dim=-1)
        expert_outs = []
        for i in range(self.num_experts):
            q, k, v = self.qkv_proj[i](xn).chunk(3, dim=-1)
            q = F.relu(q.contiguous().view(B, L, self.num_heads, -1).transpose(1, 2))
            k = F.relu(k.contiguous().view(B, L, self.num_heads, -1).transpose(1, 2))
            v =        v.contiguous().view(B, L, self.num_heads, -1).transpose(1, 2)
            kv   = torch.einsum('bnld,bnle->bnle', k, v)
            z    = 1 / (torch.einsum('bnld,bnld->bnl', q, k) + 1e-6).unsqueeze(-1)
            attn = torch.einsum('bnld,bnle->bnle', q, kv) * z
            attn = attn.transpose(1, 2).reshape(B, L, D)
            expert_outs.append(self.drop(self.out_proj[i](attn)))
        combined = sum(gate_probs[..., i:i+1] * expert_outs[i] for i in range(self.num_experts))
        x = x + self.drop(combined)
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class CALMoETransUNet(nn.Module):
    def __init__(self, in_channels=12, out_channels=2,
                 embed_dim=128, num_heads=4, mlp_dim=256,
                 transformer_depth=1, num_experts=3):
        super().__init__()
        self.enc1 = ResidualBlock(in_channels, 64)
        self.enc2 = ResidualBlock(64, 128)
        self.enc3 = ResidualBlock(128, 256)
        self.enc4 = ResidualBlock(256, embed_dim)
        self.flat = nn.Flatten(2)
        self.transformers = nn.ModuleList([
            MoEAttentionBlock(embed_dim, num_heads, mlp_dim, num_experts)
            for _ in range(transformer_depth)
        ])
        self.up4  = nn.ConvTranspose2d(embed_dim, 256, 2, stride=2)
        self.dec4 = ResidualBlock(256, 256)
        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ResidualBlock(128, 128)
        self.up2  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ResidualBlock(64, 64)
        self.head = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b, c, h, w = e4.shape
        t = self.flat(e4).permute(0, 2, 1)
        for blk in self.transformers:
            t = blk(t)
        t  = t.permute(0, 2, 1).view(b, c, h, w)
        d4 = self.dec4(self.up4(t)  + e3)
        d3 = self.dec3(self.up3(d4) + e2)
        d2 = self.dec2(self.up2(d3) + e1)
        return self.head(d2)


# ------------------------------------------------------------------
# 2.  Dataset
# ------------------------------------------------------------------

class CanadaWildfireDataset(Dataset):
    """
    Input  : 12-channel float32 tensor  [S1_pre(3), S1_post(3), S2_pre(3), S2_post(3)]
    Target : binary mask (burned=1) from 'poly' band
    Uses h5py directly for fast indexing and patch loading.
    """

    def __init__(self, h5_path, patch_size=256, patches_per_event=16,
                 events=None, augment=True):
        self.h5_path    = h5_path
        self.patch_size = patch_size
        self.augment    = augment
        self.samples    = []   # list of (group_path, y0, x0)

        groups_txt = Path(h5_path).with_name(
            Path(h5_path).stem.replace('-uint16', '') + '-groups.txt'
        )
        all_events = []
        if groups_txt.exists():
            with open(groups_txt) as f:
                for line in f:
                    line = line.strip().strip("'()\r\n ,")
                    if line.startswith('/20') and line.count('/') == 2:
                        all_events.append(line)   # e.g. /2017/CA_2017_BC_1157
        else:
            print("WARNING: groups file not found.")

        if events:
            all_events = [e for e in all_events if any(ev in e for ev in events)]

        if not all_events:
            raise ValueError("No events found. Check groups file or --event argument.")

        print(f"Indexing {len(all_events)} events (fast h5py scan) ...")
        self._build_index(all_events, patches_per_event)

    def _build_index(self, event_paths, patches_per_event):
        """Open h5py once, read only shapes — very fast."""
        skipped = 0
        with h5py.File(self.h5_path, 'r') as f:
            for ep in event_paths:
                group = ep.strip('/')   # '2017/CA_2017_BC_1157'
                try:
                    mask_ds = f[group]['mask']
                    # shape is (mask_band, y, x)
                    _, H, W = mask_ds.shape
                    if H < self.patch_size or W < self.patch_size:
                        skipped += 1
                        continue
                    for _ in range(patches_per_event):
                        y0 = random.randint(0, H - self.patch_size)
                        x0 = random.randint(0, W - self.patch_size)
                        self.samples.append((group, y0, x0))
                except Exception as e:
                    skipped += 1
        if skipped:
            print(f"  Skipped {skipped} events (too small or missing).")
        print(f"  -> {len(self.samples)} patches indexed.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        group, y0, x0 = self.samples[idx]
        P = self.patch_size

        with h5py.File(self.h5_path, 'r') as f:
            g = f[group]
            # s1: shape (time, band, y, x)  — time 0=pre, 1=post
            s1_pre  = g['s1'][0, :, y0:y0+P, x0:x0+P].astype(np.float32)  # (3,P,P)
            s1_post = g['s1'][1, :, y0:y0+P, x0:x0+P].astype(np.float32)
            s2_pre  = g['s2'][0, :, y0:y0+P, x0:x0+P].astype(np.float32)
            s2_post = g['s2'][1, :, y0:y0+P, x0:x0+P].astype(np.float32)
            # mask: shape (mask_band, y, x)  — band 0 = poly
            mask    = g['mask'][0, y0:y0+P, x0:x0+P].astype(np.int64)

        img  = np.concatenate([s1_pre, s1_post, s2_pre, s2_post], axis=0) / 65535.0
        img  = torch.from_numpy(img)
        mask = torch.from_numpy((mask > 0).astype(np.int64))

        if self.augment:
            if random.random() > 0.5:
                img  = torch.flip(img,  dims=[-1])
                mask = torch.flip(mask, dims=[-1])
            if random.random() > 0.5:
                img  = torch.flip(img,  dims=[-2])
                mask = torch.flip(mask, dims=[-2])

        return img, mask


# ------------------------------------------------------------------
# 3.  Fast Memmap Dataset (uses memory-mapped .npy — instant open, zero RAM load)
# ------------------------------------------------------------------

class MemmapDataset(Dataset):
    """
    Memory-mapped dataset from patches_images.npy + patches_masks.npy.
    Opens in milliseconds. Reads only needed patches from disk per batch.
    Create these files with: python3 preextract_patches.py
    """
    def __init__(self, images_npy, masks_npy, augment=True):
        print(f"Opening memmap: {images_npy}")
        self.images  = np.load(images_npy, mmap_mode='r')  # (N,12,P,P) float32
        self.masks   = np.load(masks_npy,  mmap_mode='r')  # (N,P,P)    int8/int64
        self.augment = augment
        print(f"  -> {len(self.images)} patches  shape={self.images.shape[1:]}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img  = torch.from_numpy(self.images[idx].copy())
        mask = torch.from_numpy(self.masks[idx].astype(np.int64))
        # Replace NaN pixels (water-masked areas in raw HDF5) with 0
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


# ------------------------------------------------------------------
# 4.  Loss & metrics
# ------------------------------------------------------------------

def dice_loss(pred, target, smooth=1.0):
    prob   = torch.softmax(pred, dim=1)[:, 1]
    target = target.float()
    inter  = (prob * target).sum()
    return 1 - (2 * inter + smooth) / (prob.sum() + target.sum() + smooth)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss, total_dice = 0.0, 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = ce(logits, masks) + dice_loss(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        with torch.no_grad():
            pred  = logits.argmax(1)
            inter = ((pred == 1) & (masks == 1)).float().sum()
            denom = (pred == 1).float().sum() + (masks == 1).float().sum() + 1e-6
            total_dice += (2 * inter / denom).item()
    n = len(loader)
    return total_loss / n, total_dice / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss   = ce(logits, masks) + dice_loss(logits, masks)
        total_loss += loss.item()
        pred  = logits.argmax(1)
        inter = ((pred == 1) & (masks == 1)).float().sum()
        union = ((pred == 1) | (masks == 1)).float().sum()
        denom = (pred == 1).float().sum() + (masks == 1).float().sum() + 1e-6
        total_dice += (2 * inter / denom).item()
        total_iou  += (inter / (union + 1e-6)).item()
    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


# ------------------------------------------------------------------
# 4.  Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train CA-LMoETransUNet on Canada wildfire dataset")
    parser.add_argument("--h5",          default="wildfire-s1s2-alos-dataset-canada-uint16.h5")
    parser.add_argument("--npz",         default="patches_ca.npz",
                        help="Pre-extracted patches (from preextract_patches.py). Used if file exists.")
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch-size",  type=int,   default=4)
    parser.add_argument("--patch-size",  type=int,   default=256)
    parser.add_argument("--patches",     type=int,   default=16,
                        help="Random patches sampled per event (only used with HDF5)")
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--val-split",   type=float, default=0.2)
    parser.add_argument("--event",       type=str,   default=None,
                        help="Single event name, e.g. CA_2017_BC_1157")
    parser.add_argument("--out-dir",     default="checkpoints_ca")
    parser.add_argument("--embed-dim",   type=int,   default=128)
    parser.add_argument("--num-experts", type=int,   default=3)
    parser.add_argument("--depth",       type=int,   default=1)
    args = parser.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print("\n-- Building dataset --------------------------")
    images_npy = Path("patches_images.npy")
    masks_npy  = Path("patches_masks.npy")
    if images_npy.exists() and masks_npy.exists():
        full_ds = MemmapDataset(images_npy, masks_npy)
    elif Path(args.npz).exists():
        print(f"Found npz (slow to load). Converting to npy first is recommended.")
        print(f"Loading {args.npz} ...")
        data = np.load(args.npz)
        import tempfile, os
        full_ds = MemmapDataset.__new__(MemmapDataset)
        full_ds.images  = data['images']
        full_ds.masks   = data['masks']
        full_ds.augment = True
    else:
        print("No .npy or .npz found, loading from HDF5. Run preextract_patches.py first!")
        events = [args.event] if args.event else None
        full_ds = CanadaWildfireDataset(
            h5_path=args.h5,
            patch_size=args.patch_size,
            patches_per_event=args.patches,
            events=events,
        )

    val_size   = max(1, int(len(full_ds) * args.val_split))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    print(f"  Train patches: {train_size}  |  Val patches: {val_size}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    print("\n-- Building model ----------------------------")
    model = CALMoETransUNet(
        in_channels=12, out_channels=2,
        embed_dim=args.embed_dim, num_heads=4,
        mlp_dim=args.embed_dim * 2,
        transformer_depth=args.depth,
        num_experts=args.num_experts,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_dice = 0.0
    print("\n-- Training ----------------------------------")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_dice           = train_one_epoch(model, train_loader, optimizer, device)
        vl_loss, vl_dice, vl_iou  = evaluate(model, val_loader, device)
        scheduler.step()
        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"train loss={tr_loss:.4f} dice={tr_dice:.4f}  |  "
              f"val loss={vl_loss:.4f} dice={vl_dice:.4f} IoU={vl_iou:.4f}")
        if vl_dice > best_dice:
            best_dice = vl_dice
            ckpt_path = out_dir / "best_model.pth"
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_dice":    vl_dice,
                "val_iou":     vl_iou,
                "args":        vars(args),
            }, ckpt_path)
            print(f"  Saved best model -> {ckpt_path}  (dice={best_dice:.4f})")

    print(f"\nDone. Best val Dice: {best_dice:.4f}")
    print(f"Checkpoint: {out_dir.resolve()}/best_model.pth")


if __name__ == "__main__":
    main()

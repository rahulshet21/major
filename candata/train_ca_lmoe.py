#!/usr/bin/env python3
"""
Train CA-LMoETransUNet on Canada wildfire S1/S2 dataset or legacy dataset using Unified Loader.

Model Architecture: CA-LMoETransUNet (CNN encoder -> Transformer + MoE bottleneck -> CNN decoder)
Channels: 12 (s1_pre, s1_post, s2_pre, s2_post)
Target: Binary Burned Area Mask (0 = background, 1 = burned)

Usage:
  python3 train_ca_lmoe.py [--h5 PATH] [--epochs 20] [--batch-size 4] [--patch-size 256]
                           [--lr 1e-4] [--device cuda/mps/cpu]
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from unified_dataset import (
    CanadaHDF5Dataset,
    LegacyNpyDataset,
    MemmapDataset,
)

# ------------------------------------------------------------------
# 1. Model Architecture (CA-LMoETransUNet - DO NOT ALTER ARCHITECTURE)
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
# 2. Helpers & Loss
# ------------------------------------------------------------------

def get_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    n = len(loader) if len(loader) > 0 else 1
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
    n = len(loader) if len(loader) > 0 else 1
    return total_loss / n, total_dice / n, total_iou / n


# ------------------------------------------------------------------
# 3. Main Training Execution
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train CA-LMoETransUNet on Canada/Legacy wildfire dataset")
    parser.add_argument("--h5",          default="candata/wildfire-s1s2-alos-dataset-canada-uint16.h5", help="HDF5 file path")
    parser.add_argument("--npz",         default="patches_ca.npz", help="Pre-extracted .npz patches")
    parser.add_argument("--images-npy",  default="patches_images.npy", help="Pre-extracted images .npy")
    parser.add_argument("--masks-npy",   default="patches_masks.npy", help="Pre-extracted masks .npy")
    parser.add_argument("--legacy-dir",  default="wildfire/processed", help="Directory of legacy 30-channel .npy files")
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch-size",  type=int,   default=4)
    parser.add_argument("--patch-size",  type=int,   default=256)
    parser.add_argument("--patches",     type=int,   default=16, help="Patches per event when sampling HDF5")
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--val-split",   type=float, default=0.2)
    parser.add_argument("--event",       type=str,   default=None, help="Filter to single event")
    parser.add_argument("--out-dir",     default="checkpoints_ca", help="Checkpoint output folder")
    parser.add_argument("--embed-dim",   type=int,   default=128)
    parser.add_argument("--num-experts", type=int,   default=3)
    parser.add_argument("--depth",       type=int,   default=1)
    parser.add_argument("--device",      type=str,   default=None, help="Device (cuda, mps, cpu)")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device(args.device)
    print(f"Device fallback selected: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n-- Building Unified Dataset --------------------------")
    images_npy = Path(args.images_npy)
    masks_npy  = Path(args.masks_npy)
    npz_path   = Path(args.npz)
    legacy_dir = Path(args.legacy_dir)

    if images_npy.exists() and masks_npy.exists():
        print(f"Loading memory-mapped dataset: {images_npy} and {masks_npy}")
        full_ds = MemmapDataset(images_npy, masks_npy)
    elif npz_path.exists():
        print(f"Loading .npz cached dataset: {npz_path}")
        data = np.load(npz_path)
        full_ds = MemmapDataset.__new__(MemmapDataset)
        full_ds.images  = data['images']
        full_ds.masks   = data['masks']
        full_ds.augment = True
    elif legacy_dir.exists() and len(list(legacy_dir.glob('*_image.npy'))) > 0:
        print(f"Loading legacy processed dataset from: {legacy_dir}")
        full_ds = LegacyNpyDataset(
            processed_dir=legacy_dir,
            patch_size=args.patch_size,
            patches_per_event=args.patches,
            event_list=[args.event] if args.event else None,
        )
    else:
        h5_file = Path(args.h5)
        if not h5_file.exists() and (Path("candata") / h5_file).exists():
            h5_file = Path("candata") / h5_file
        print(f"Loading direct HDF5 dataset: {h5_file}")
        full_ds = CanadaHDF5Dataset(
            h5_path=h5_file,
            patch_size=args.patch_size,
            patches_per_event=args.patches,
            events=[args.event] if args.event else None,
        )

    val_size   = max(1, int(len(full_ds) * args.val_split))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    print(f"  Train patches: {train_size}  |  Val patches: {val_size}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("\n-- Building CA-LMoETransUNet Model ----------------------------")
    model = CALMoETransUNet(
        in_channels=12,
        out_channels=2,
        embed_dim=args.embed_dim,
        num_heads=4,
        mlp_dim=args.embed_dim * 2,
        transformer_depth=args.depth,
        num_experts=args.num_experts,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_dice = 0.0
    print("\n-- Training Loop ----------------------------------")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_dice          = train_one_epoch(model, train_loader, optimizer, device)
        vl_loss, vl_dice, vl_iou = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch:03d}/{args.epochs:03d}  "
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

    print(f"\nDone! Best Validation Dice: {best_dice:.4f}")
    print(f"Best checkpoint: {out_dir.resolve()}/best_model.pth")


if __name__ == "__main__":
    main()

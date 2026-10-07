#!/usr/bin/env python3
"""
Model Zoo for Burned-Area Model Comparison.

Implements 5 alternative architectures and a factory function `get_model(name)`.
All neural models accept [B, 12, H, W] input and return [B, 2, H, W] logits.

Models:
  1. unet      – smp.Unet (ResNet-34 encoder)
  2. deeplab   – smp.DeepLabV3Plus (ResNet-50 encoder)
  3. fcsiam    – FC-Siam-diff (hand-written, shared-weight Siamese encoder)
  4. segformer – smp.Segformer or HuggingFace fallback
  5. rf        – RandomForest / XGBoost per-pixel classifier (not a nn.Module)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# 1 & 2.  SMP-based models  (Unet, DeepLabV3Plus)
# ──────────────────────────────────────────────────────────────────────
def _smp_unet(in_channels=12):
    import segmentation_models_pytorch as smp
    return smp.Unet(
        "resnet34", encoder_weights=None, in_channels=in_channels, classes=2
    )


def _smp_deeplab(in_channels=12):
    import segmentation_models_pytorch as smp
    return smp.DeepLabV3Plus(
        "resnet50", encoder_weights=None, in_channels=in_channels, classes=2
    )


# ──────────────────────────────────────────────────────────────────────
# 3.  FC-Siam-diff  (hand-written, ~80 lines)
# ──────────────────────────────────────────────────────────────────────
class _ConvBlock(nn.Module):
    """Two 3×3 convs + BN + ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FCSiamDiff(nn.Module):
    """
    FC-Siam-diff for change detection.

    Input [B, 12, H, W] is split:
        pre  = channels [0,1,2, 6,7,8]   (S1 pre + S2 pre)
        post = channels [3,4,5, 9,10,11] (S1 post + S2 post)

    A shared-weight encoder (4 stages) processes both halves.
    The decoder fuses |feat_pre − feat_post| at every skip connection.
    Output [B, 2, H, W].
    """
    PRE_IDX  = [0, 1, 2, 6, 7, 8]
    POST_IDX = [3, 4, 5, 9, 10, 11]

    def __init__(self, in_ch=6, base_ch=64, out_ch=2):
        super().__init__()
        # Shared encoder
        self.enc1 = _ConvBlock(in_ch, base_ch)
        self.enc2 = _ConvBlock(base_ch, base_ch * 2)
        self.enc3 = _ConvBlock(base_ch * 2, base_ch * 4)
        self.enc4 = _ConvBlock(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool2d(2)

        # Decoder  (input = upsampled + diff-skip)
        self.up4  = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec4 = _ConvBlock(base_ch * 4 + base_ch * 4, base_ch * 4)   # diff-skip
        self.up3  = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec3 = _ConvBlock(base_ch * 2 + base_ch * 2, base_ch * 2)
        self.up2  = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec2 = _ConvBlock(base_ch + base_ch, base_ch)
        self.head = nn.Conv2d(base_ch, out_ch, 1)

    def _encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        return e1, e2, e3, e4

    def forward(self, x):
        half_ch = x.shape[1] // 2
        # For 12-channel candata: PRE_IDX and POST_IDX works
        if x.shape[1] == 12:
            pre  = x[:, self.PRE_IDX]
            post = x[:, self.POST_IDX]
        else:
            # Just split in half
            pre = x[:, :half_ch]
            post = x[:, half_ch:]

        e1_a, e2_a, e3_a, e4_a = self._encode(pre)
        e1_b, e2_b, e3_b, e4_b = self._encode(post)

        # Bottleneck (average of both encodings)
        d = self.up4(e4_a)
        d = self.dec4(torch.cat([d, torch.abs(e3_a - e3_b)], dim=1))
        d = self.up3(d)
        d = self.dec3(torch.cat([d, torch.abs(e2_a - e2_b)], dim=1))
        d = self.up2(d)
        d = self.dec2(torch.cat([d, torch.abs(e1_a - e1_b)], dim=1))
        return self.head(d)


# ──────────────────────────────────────────────────────────────────────
# 4.  SegFormer  (smp first, HuggingFace fallback)
# ──────────────────────────────────────────────────────────────────────
class _HFSegformerWrapper(nn.Module):
    """Wraps HuggingFace SegformerForSemanticSegmentation to match our API."""
    def __init__(self, in_channels=12):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation, SegformerConfig
        cfg = SegformerConfig(
            num_channels=in_channels,
            num_labels=2,
            num_encoder_blocks=4,
            depths=[2, 2, 2, 2],
            hidden_sizes=[32, 64, 160, 256],
            decoder_hidden_size=256,
        )
        self.model = SegformerForSemanticSegmentation(cfg)
        # Monkey-patch all .view() to .reshape() to fix MPS backward() crash
        self._patch_view_to_reshape(self.model)

    @staticmethod
    def _patch_view_to_reshape(module):
        """Recursively replace Tensor.view calls by patching forward methods."""
        # The MPS issue is inside HF internals; easiest fix is contiguous() before view.
        # We handle this in forward() by moving to CPU on MPS.
        pass

    def forward(self, x):
        # HF SegFormer's attention uses .view() in a way that breaks MPS backward.
        # Move to CPU for the forward+backward pass when on MPS.
        device = x.device
        if device.type == 'mps':
            x = x.cpu()
            self.model.cpu()
        out = self.model(pixel_values=x)
        logits = out.logits  # [B, 2, H/4, W/4] typically
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:],
                                   mode='bilinear', align_corners=False)
        if device.type == 'mps':
            logits = logits.to(device)
            self.model.to(device)
        return logits


def _segformer(in_channels=12):
    """Try smp.Segformer(mit_b0) first for speed; fall back to HuggingFace."""
    try:
        import segmentation_models_pytorch as smp
        model_cls = getattr(smp, "Segformer", None)
        if model_cls is None:
            raise AttributeError("smp has no Segformer")
        # mit_b0 is lighter; mit_b1 may not have pretrained=None support in all smp versions
        for backbone in ["mit_b0", "mit_b1"]:
            try:
                m = model_cls(backbone, encoder_weights=None,
                              in_channels=in_channels, classes=2)
                print(f"  [segformer] Using smp.Segformer({backbone})")
                return m
            except Exception:
                continue
        raise RuntimeError("No smp Segformer backbone worked")
    except Exception as e:
        print(f"  [segformer] smp.Segformer failed ({e}), "
              "falling back to HuggingFace SegformerForSemanticSegmentation")
        return _HFSegformerWrapper(in_channels=in_channels)


# ──────────────────────────────────────────────────────────────────────
# 5.  Random Forest / XGBoost pixel classifier
# ──────────────────────────────────────────────────────────────────────
def compute_rf_features(img):
    """
    Compute per-pixel features from a [12, H, W] normalised image.

    Returns feature array [H*W, n_features] and feature names list.
    Channel indices:
        0-2: S1 pre (ND, VH, VV)   3-5: S1 post (ND, VH, VV)
        6-8: S2 pre (B4, B8, B12)  9-11: S2 post (B4, B8, B12)
    """
    C, H, W = img.shape
    flat = img.reshape(C, -1).T  # [N, 12]

    eps = 1e-8
    # NBR_pre = (B8 - B12) / (B8 + B12)  using ch 7, 8
    nbr_pre = (flat[:, 7] - flat[:, 8]) / (flat[:, 7] + flat[:, 8] + eps)
    # NBR_post using ch 10, 11
    nbr_post = (flat[:, 10] - flat[:, 11]) / (flat[:, 10] + flat[:, 11] + eps)
    dnbr   = nbr_pre - nbr_post
    dnir   = flat[:, 10] - flat[:, 7]
    dswir  = flat[:, 11] - flat[:, 8]
    dred   = flat[:, 9]  - flat[:, 6]
    dvh    = flat[:, 4]  - flat[:, 1]
    dvv    = flat[:, 5]  - flat[:, 2]

    extras = np.column_stack([nbr_pre, nbr_post, dnbr,
                              dnir, dswir, dred, dvh, dvv])
    features = np.concatenate([flat, extras], axis=1)

    feature_names = [
        "s1pre_ND", "s1pre_VH", "s1pre_VV",
        "s1post_ND", "s1post_VH", "s1post_VV",
        "s2pre_B4", "s2pre_B8", "s2pre_B12",
        "s2post_B4", "s2post_B8", "s2post_B12",
        "NBR_pre", "NBR_post", "dNBR",
        "dNIR", "dSWIR", "dRed", "dVH", "dVV",
    ]
    return features.astype(np.float32), feature_names


# ──────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────
NEURAL_MODELS = ["unet", "deeplab", "fcsiam", "segformer"]
ALL_MODELS = NEURAL_MODELS + ["rf"]


def get_model(name, in_channels=12):
    """
    Factory that returns an nn.Module for the given model name.
    For 'rf' this returns None (handled separately).
    """
    name = name.lower()
    if name == "unet":
        return _smp_unet(in_channels=in_channels)
    elif name == "deeplab":
        return _smp_deeplab(in_channels=in_channels)
    elif name == "fcsiam":
        # FCSiamDiff splits channels in half.
        return FCSiamDiff(in_ch=in_channels // 2, out_ch=2)
    elif name == "segformer":
        return _segformer(in_channels=in_channels)
    elif name == "rf":
        return None  # not an nn.Module
    else:
        raise ValueError(f"Unknown model name: {name}. "
                         f"Choose from {ALL_MODELS}")


def count_params(model):
    """Count trainable parameters of an nn.Module (returns 0 for None)."""
    if model is None:
        return 0
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

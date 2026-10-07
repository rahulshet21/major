"""
metrics.py — Dice/IoU/Precision/Recall/Confusion, PR/ROC, best threshold, ensemble.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
from config import CURVE_SAMPLE_SIZE


def _flatten(pred: np.ndarray, gt: np.ndarray):
    p = pred.ravel().astype(bool)
    g = gt.ravel().astype(bool)
    return p, g


def confusion(pred_bin: np.ndarray, gt: np.ndarray) -> dict:
    p, g = _flatten(pred_bin, gt)
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    tn = int((~p & ~g).sum())
    total = tp + fp + fn + tn
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, total=total)


def compute_metrics(pred_bin: np.ndarray, gt: np.ndarray) -> dict:
    c = confusion(pred_bin, gt)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    precision  = tp / (tp + fp + 1e-8)
    recall     = tp / (tp + fn + 1e-8)
    dice       = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou        = tp / (tp + fp + fn + 1e-8)
    return dict(dice=dice, iou=iou, precision=precision, recall=recall, **c)


def compute_metrics_at_threshold(prob: np.ndarray, gt: np.ndarray, threshold: float) -> dict:
    pred_bin = (prob > threshold).astype(np.uint8)
    return compute_metrics(pred_bin, gt)


def best_f1_threshold(prob: np.ndarray, gt: Optional[np.ndarray],
                      n_thresholds: int = 51) -> float:
    """Sweep thresholds and return the one with best Dice/F1."""
    if gt is None:
        return 0.5
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    best_t, best_dice = 0.5, 0.0
    for t in thresholds:
        m = compute_metrics_at_threshold(prob, gt, t)
        if m["dice"] > best_dice:
            best_dice = m["dice"]
            best_t = float(t)
    return best_t


def pr_roc_curves(prob: np.ndarray, gt: np.ndarray, n_thresholds: int = 51):
    """Return (thresholds, precisions, recalls, fprs) arrays."""
    # Stratified sample to cap computation
    flat_p = prob.ravel()
    flat_g = gt.ravel().astype(np.uint8)
    if len(flat_p) > CURVE_SAMPLE_SIZE:
        idx = np.random.default_rng(0).choice(len(flat_p), CURVE_SAMPLE_SIZE, replace=False)
        flat_p = flat_p[idx]
        flat_g = flat_g[idx]

    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    precisions, recalls, fprs = [], [], []
    for t in thresholds:
        pred = flat_p > t
        g = flat_g.astype(bool)
        tp = (pred & g).sum()
        fp = (pred & ~g).sum()
        fn = (~pred & g).sum()
        tn = (~pred & ~g).sum()
        precisions.append(tp / (tp + fp + 1e-8))
        recalls.append(tp / (tp + fn + 1e-8))
        fprs.append(fp / (fp + tn + 1e-8))
    return np.array(thresholds), np.array(precisions), np.array(recalls), np.array(fprs)


def dice_vs_threshold(prob: np.ndarray, gt: np.ndarray, n_thresholds: int = 51):
    """Return (thresholds, dice_scores)."""
    flat_p = prob.ravel()
    flat_g = gt.ravel().astype(np.uint8)
    if len(flat_p) > CURVE_SAMPLE_SIZE:
        idx = np.random.default_rng(0).choice(len(flat_p), CURVE_SAMPLE_SIZE, replace=False)
        flat_p = flat_p[idx]
        flat_g = flat_g[idx]
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    dices = []
    for t in thresholds:
        pred = (flat_p > t).astype(np.uint8)
        g    = flat_g
        tp = ((pred == 1) & (g == 1)).sum()
        fp = ((pred == 1) & (g == 0)).sum()
        fn = ((pred == 0) & (g == 1)).sum()
        dices.append(2 * tp / (2 * tp + fp + fn + 1e-8))
    return np.array(thresholds), np.array(dices)


def ensemble_prob(probs: list[np.ndarray]) -> np.ndarray:
    """Mean probability ensemble."""
    arr = np.stack([p for p in probs if p is not None], axis=0)
    return arr.mean(axis=0)


def ensemble_majority_vote(probs: list[np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """Majority vote across models."""
    masks = [(p > threshold).astype(np.uint8) for p in probs if p is not None]
    arr = np.stack(masks, axis=0)
    return (arr.mean(axis=0) >= 0.5).astype(np.uint8)


def agreement_map(probs: list[np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """Returns int array [0, n_models] = how many models say burned."""
    votes = np.stack([(p > threshold).astype(np.uint8) for p in probs if p is not None], axis=0)
    return votes.sum(axis=0)


def make_error_map_rgba(pred_bin: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """RGBA [H,W,4] uint8. Green=TP, Red=FP, Blue=FN, Gray=TN."""
    H, W = pred_bin.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    tp = (pred_bin == 1) & (gt == 1)
    fp = (pred_bin == 1) & (gt == 0)
    fn = (pred_bin == 0) & (gt == 1)
    tn = (pred_bin == 0) & (gt == 0)
    rgba[tp] = [50,  200,  80, 200]  # Green
    rgba[fp] = [220,  50,  50, 200]  # Red
    rgba[fn] = [ 50,  80, 220, 200]  # Blue
    rgba[tn] = [100, 100, 100,  60]  # Gray, low alpha
    return rgba


def burned_area_ha(mask: np.ndarray, pixel_size_m: float = 10.0) -> float:
    return float(mask.sum()) * pixel_size_m * pixel_size_m / 10_000.0


def build_leaderboard(metrics_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Return sorted leaderboard DataFrame."""
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame()
    df = metrics_df.copy()
    for col in ["dice", "iou", "precision", "recall"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "dice" in df.columns:
        df = df.sort_values("dice", ascending=False)
    return df.reset_index(drop=True)

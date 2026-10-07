#!/usr/bin/env python3
"""
Prediction & comparison pipeline for model-comparison feature.

Runs 5 models + the existing CA-LMoETransUNet on a given event, saves
per-model predictions, a combined comparison image, and a metrics CSV.

Usage:
  python3 predict_compare.py --input_dir synthetic_test --weights best_model.pth
  python3 predict_compare.py --h5 candata/wildfire-s1s2-alos-dataset-canada-uint16.h5 \\
      --event CA_2017_BC_1157 --weights best_model.pth
"""

import argparse
import csv
import os
import time
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import from existing project modules (read-only)
from train_ca_lmoe import CALMoETransUNet
from unified_dataset import (
    load_event_from_h5, clean_and_normalize, resolve_existing_path,
)
from predict_folder import prepare_folder_input, predict_tiled

# Import from our new modules
from models_compare import (
    get_model, count_params, compute_rf_features,
    NEURAL_MODELS, ALL_MODELS,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def get_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def safe_output_path(path):
    """If path already exists, insert a timestamp before the extension."""
    p = Path(path)
    if p.exists():
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        p = p.with_stem(f"{p.stem}_{ts}")
    return p


def compute_metrics(pred_binary, gt_mask):
    """Compute dice, iou, precision, recall from binary arrays."""
    pred = pred_binary.astype(bool).ravel()
    gt   = gt_mask.astype(bool).ravel()

    tp = (pred & gt).sum()
    fp = (pred & ~gt).sum()
    fn = (~pred & gt).sum()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    dice      = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou       = tp / (tp + fp + fn + 1e-8)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


# ------------------------------------------------------------------
# Per-model inference
# ------------------------------------------------------------------

def predict_neural(model, img, device, patch_size=256, overlap=32):
    """Run tiled prediction for a neural model. Returns prob [H, W]."""
    model.to(device)
    model.eval()
    return predict_tiled(model, img, device,
                         patch_size=patch_size, overlap=overlap)


def predict_rf(clf, img, chunk_size=100_000):
    """Run random-forest prediction. Returns prob [H, W]."""
    C, H, W = img.shape
    features, _ = compute_rf_features(img)
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

    # Predict in chunks to limit memory
    probs = np.zeros(H * W, dtype=np.float32)
    for start in range(0, len(features), chunk_size):
        end = min(start + chunk_size, len(features))
        chunk = features[start:end]
        if hasattr(clf, "predict_proba"):
            p = clf.predict_proba(chunk)
            # Burned class probability (class index 1)
            if p.shape[1] >= 2:
                probs[start:end] = p[:, 1]
            else:
                probs[start:end] = p[:, 0]
        else:
            probs[start:end] = clf.predict(chunk).astype(np.float32)

    return probs.reshape(H, W)


def load_neural_model(model_name, ckpt_dir, device):
    """Load a trained neural model from checkpoint."""
    ckpt_path = Path(ckpt_dir) / f"{model_name}_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = get_model(model_name)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = checkpoint.get("model_state", checkpoint)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_rf_model(ckpt_dir):
    """Load a trained RF/XGB model from checkpoint."""
    import joblib
    path = Path(ckpt_dir) / "rf_best.joblib"
    if not path.exists():
        raise FileNotFoundError(f"RF checkpoint not found: {path}")
    return joblib.load(path)


def load_calmoe(weights_path, device):
    """Load the existing CA-LMoETransUNet model (read-only)."""
    weights = resolve_existing_path(weights_path)
    checkpoint = torch.load(weights, map_location="cpu")
    ckpt_args = checkpoint.get("args", {})
    model = CALMoETransUNet(
        in_channels=12, out_channels=2,
        embed_dim=ckpt_args.get("embed_dim", 128),
        num_experts=ckpt_args.get("num_experts", 3),
        transformer_depth=ckpt_args.get("depth", 1),
    )
    state = checkpoint.get("model_state", checkpoint)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ------------------------------------------------------------------
# Visualization  (per-model 4-panel + combined comparison)
# ------------------------------------------------------------------

def save_single_prediction_png(out_path, img, gt_mask, prob, threshold,
                                event_name, model_name):
    """4-panel per-model prediction PNG."""
    rgb = make_rgb(img)
    pred_bin = (prob > threshold).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(rgb)
    axes[0].set_title(f"S2 Pre RGB\n{event_name}")
    axes[0].axis("off")

    if gt_mask is not None:
        axes[1].imshow(gt_mask, cmap="gray")
        axes[1].set_title("Ground Truth")
    else:
        axes[1].text(0.5, 0.5, "No GT", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=14)
        axes[1].set_title("Ground Truth (N/A)")
    axes[1].axis("off")

    axes[2].imshow(prob, cmap="inferno", vmin=0, vmax=1)
    axes[2].set_title(f"{model_name}\nProbability")
    axes[2].axis("off")

    axes[3].imshow(rgb)
    overlay = np.where(pred_bin == 1, 1.0, np.nan)
    axes[3].imshow(overlay, cmap="Reds", alpha=0.6, vmin=0, vmax=1)
    axes[3].set_title(f"{model_name}\nOverlay")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(safe_output_path(out_path), dpi=150)
    plt.close()


def make_rgb(img):
    """S2 pre-fire false-colour RGB from 12-ch image."""
    r = img[6]
    g = img[7]
    b = img[8]
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 3.0, 0, 1)


def make_rgb_post(img):
    """S2 post-fire false-colour RGB from 12-ch image."""
    r = img[9]
    g = img[10]
    b = img[11]
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 3.0, 0, 1)


def make_error_map(pred_bin, gt_mask):
    """
    Error map: FP=red, FN=blue, correct=grey.
    Returns RGBA [H, W, 4].
    """
    H, W = pred_bin.shape
    err = np.zeros((H, W, 4), dtype=np.float32)

    tp = (pred_bin == 1) & (gt_mask == 1)
    tn = (pred_bin == 0) & (gt_mask == 0)
    fp = (pred_bin == 1) & (gt_mask == 0)
    fn = (pred_bin == 0) & (gt_mask == 1)

    # Correct → grey with slight alpha
    err[tp | tn] = [0.5, 0.5, 0.5, 0.15]
    # TP → faint green
    err[tp] = [0.2, 0.8, 0.2, 0.4]
    # FP → red
    err[fp] = [1.0, 0.0, 0.0, 0.7]
    # FN → blue
    err[fn] = [0.0, 0.0, 1.0, 0.7]

    return err


def save_combined_comparison(out_path, img, gt_mask, results_dict,
                              event_name, threshold):
    """
    Combined comparison image.
    Row 0: S2 pre RGB, S2 post RGB, Ground truth
    Row 1..N: pairs of (prediction, error map) per model
    """
    n_models = len(results_dict)
    # Layout: 3 columns for header + 2 cols per model row (pred + error)
    # Actually: row0 = 3 panels header; subsequent rows = one per model, 2 panels each
    # Let's do a grid: first row = 3 cols, then each model gets 2 cols in 3-col grid
    # Better: 3 columns throughout.  Row0 = rgb_pre, rgb_post, gt.
    # Then each model row = prediction, error_map, (metrics text or blank)

    n_rows = 1 + n_models
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))

    # Ensure axes is 2D
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # --- Row 0: header ---
    rgb_pre = make_rgb(img)
    rgb_post = make_rgb_post(img)

    axes[0, 0].imshow(rgb_pre)
    axes[0, 0].set_title("S2 Pre-Fire RGB", fontsize=11, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(rgb_post)
    axes[0, 1].set_title("S2 Post-Fire RGB", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")

    if gt_mask is not None:
        axes[0, 2].imshow(gt_mask, cmap="gray")
        axes[0, 2].set_title("Ground Truth", fontsize=11, fontweight="bold")
    else:
        axes[0, 2].text(0.5, 0.5, "No Ground Truth", ha="center", va="center",
                        transform=axes[0, 2].transAxes, fontsize=13)
        axes[0, 2].set_title("Ground Truth (N/A)", fontsize=11, fontweight="bold")
    axes[0, 2].axis("off")

    # --- Model rows ---
    for row_i, (model_label, info) in enumerate(results_dict.items(), start=1):
        prob = info["prob"]
        pred_bin = (prob > threshold).astype(np.uint8)
        metrics = info.get("metrics")

        # Column 0: prediction overlay
        axes[row_i, 0].imshow(rgb_pre)
        overlay = np.where(pred_bin == 1, 1.0, np.nan)
        axes[row_i, 0].imshow(overlay, cmap="Reds", alpha=0.6, vmin=0, vmax=1)
        title = model_label
        if metrics:
            title += f"\nDice={metrics['dice']:.3f}  IoU={metrics['iou']:.3f}"
        axes[row_i, 0].set_title(title, fontsize=10, fontweight="bold")
        axes[row_i, 0].axis("off")

        # Column 1: error map (only if GT exists)
        if gt_mask is not None:
            err_map = make_error_map(pred_bin, gt_mask)
            axes[row_i, 1].imshow(rgb_pre, alpha=0.3)
            axes[row_i, 1].imshow(err_map)
            axes[row_i, 1].set_title(
                f"Error Map\n(Red=FP, Blue=FN, Green=TP)", fontsize=9)
        else:
            axes[row_i, 1].imshow(prob, cmap="inferno", vmin=0, vmax=1)
            axes[row_i, 1].set_title("Probability Heatmap", fontsize=9)
        axes[row_i, 1].axis("off")

        # Column 2: probability heatmap
        im = axes[row_i, 2].imshow(prob, cmap="inferno", vmin=0, vmax=1)
        axes[row_i, 2].set_title("Probability", fontsize=9)
        axes[row_i, 2].axis("off")

    plt.suptitle(f"Model Comparison — {event_name}",
                 fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()
    plt.savefig(safe_output_path(out_path), dpi=200, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run all comparison models on a single event"
    )
    parser.add_argument("--input_dir", default=None,
                        help="Folder with S1/S2 pre/post .npy or .tif files")
    parser.add_argument("--h5", default=None, help="HDF5 file path")
    parser.add_argument("--event", default=None, help="Event name in HDF5")
    parser.add_argument("--weights", default="best_model.pth",
                        help="Path to existing CA-LMoETransUNet weights")
    parser.add_argument("--ckpt-dir", default="checkpoints_compare",
                        help="Directory with comparison model checkpoints")
    parser.add_argument("--out-dir", default="results",
                        help="Output directory for predictions")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--patch", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.input_dir is None and args.event is None:
        parser.error("Specify --input_dir <folder> or --h5 <file> --event <name>")

    device = get_device(args.device)
    print(f"Device: {device}")

    # ── Load input data ──────────────────────────────────────────
    if args.event:
        h5_path = resolve_existing_path(args.h5 or
            "candata/wildfire-s1s2-alos-dataset-canada-uint16.h5")
        print(f"Loading event {args.event} from HDF5: {h5_path}")
        img, gt_mask = load_event_from_h5(h5_path, args.event)
        event_name = args.event.split("/")[-1]
    else:
        print(f"Loading from folder: {args.input_dir}")
        img, gt_mask, _, _ = prepare_folder_input(args.input_dir)
        event_name = Path(args.input_dir).name
    out_base = Path(args.out_dir) / event_name
    out_dir = out_base
    counter = 2
    while out_dir.exists():
        out_dir = out_base.with_name(f"{out_base.name}_v{counter}")
        counter += 1
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Image shape: {img.shape}  Event: {event_name}")
    if gt_mask is not None:
        print(f"Ground truth: {gt_mask.shape}  burned px: {(gt_mask==1).sum()}")

    # ── Process each comparison model ────────────────────────────
    all_results = {}   # label → {"prob": ..., "metrics": ...}
    metrics_rows = []

    for model_name in args.models:
        print(f"\n{'─'*50}")
        print(f"  Model: {model_name}")
        print(f"{'─'*50}")

        t0 = time.time()
        try:
            if model_name == "rf":
                clf = load_rf_model(args.ckpt_dir)
                prob = predict_rf(clf, img)
                n_params = "N/A"
            else:
                model = load_neural_model(model_name, args.ckpt_dir, device)
                n_params = count_params(model)
                prob = predict_neural(model, img, device,
                                      patch_size=args.patch,
                                      overlap=args.overlap)
                # Free GPU memory
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except FileNotFoundError as e:
            print(f"  ⚠ Skipping {model_name}: {e}")
            continue

        elapsed = time.time() - t0
        pred_bin = (prob > args.threshold).astype(np.uint8)

        # Metrics
        metrics = None
        if gt_mask is not None:
            metrics = compute_metrics(pred_bin, gt_mask)
            print(f"  Dice={metrics['dice']:.4f}  IoU={metrics['iou']:.4f}  "
                  f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  "
                  f"({elapsed:.1f}s)")
        else:
            print(f"  No GT — inference time: {elapsed:.1f}s")

        # Save per-model outputs
        prob_path = safe_output_path(out_dir / f"cmp_{event_name}_{model_name}_prob.npy")
        mask_path = safe_output_path(out_dir / f"cmp_{event_name}_{model_name}_mask.npy")
        png_path  = safe_output_path(out_dir / f"cmp_{event_name}_{model_name}_prediction.png")

        np.save(prob_path, prob)
        np.save(mask_path, pred_bin)
        save_single_prediction_png(
            png_path, img, gt_mask, prob, args.threshold,
            event_name, model_name,
        )
        print(f"  Saved: {prob_path.name}, {mask_path.name}, {png_path.name}")

        all_results[model_name] = {"prob": prob, "metrics": metrics}
        metrics_rows.append({
            "model": model_name,
            "params": n_params,
            "dice": metrics["dice"] if metrics else "",
            "iou": metrics["iou"] if metrics else "",
            "precision": metrics["precision"] if metrics else "",
            "recall": metrics["recall"] if metrics else "",
            "inference_seconds": f"{elapsed:.2f}",
        })

    # ── Existing model (CA-LMoETransUNet) ────────────────────────
    print(f"\n{'─'*50}")
    print(f"  Model: CA-LMoETransUNet (existing)")
    print(f"{'─'*50}")

    # Try to load existing prediction
    existing_prob_path = out_dir / f"{event_name}_prob.npy"
    if existing_prob_path.exists():
        print(f"  Loading existing prediction: {existing_prob_path}")
        calmoe_prob = np.load(existing_prob_path)
        calmoe_elapsed = 0.0
    else:
        print(f"  Running inference from {args.weights} (read-only)")
        t0 = time.time()
        calmoe_model = load_calmoe(args.weights, device)
        calmoe_prob = predict_neural(
            calmoe_model, img, device,
            patch_size=args.patch, overlap=args.overlap,
        )
        calmoe_elapsed = time.time() - t0
        del calmoe_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Save as cmp_ prefixed (don't overwrite existing)
        calmoe_save_path = safe_output_path(
            out_dir / f"cmp_{event_name}_calmoe_prob.npy"
        )
        np.save(calmoe_save_path, calmoe_prob)
        print(f"  Saved: {calmoe_save_path.name}")

    calmoe_bin = (calmoe_prob > args.threshold).astype(np.uint8)
    calmoe_metrics = None
    calmoe_model_obj = load_calmoe(args.weights, device)
    calmoe_params = count_params(calmoe_model_obj)
    del calmoe_model_obj
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if gt_mask is not None:
        calmoe_metrics = compute_metrics(calmoe_bin, gt_mask)
        print(f"  Dice={calmoe_metrics['dice']:.4f}  "
              f"IoU={calmoe_metrics['iou']:.4f}  "
              f"P={calmoe_metrics['precision']:.4f}  "
              f"R={calmoe_metrics['recall']:.4f}")

    all_results["CA-LMoETransUNet (existing)"] = {
        "prob": calmoe_prob, "metrics": calmoe_metrics,
    }
    metrics_rows.append({
        "model": "calmoe (existing)",
        "params": calmoe_params,
        "dice": calmoe_metrics["dice"] if calmoe_metrics else "",
        "iou": calmoe_metrics["iou"] if calmoe_metrics else "",
        "precision": calmoe_metrics["precision"] if calmoe_metrics else "",
        "recall": calmoe_metrics["recall"] if calmoe_metrics else "",
        "inference_seconds": f"{calmoe_elapsed:.2f}",
    })

    # ── Combined comparison image ────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  Creating combined comparison image ...")
    print(f"{'─'*50}")

    cmp_png_path = safe_output_path(
        out_dir / f"cmp_{event_name}_ALL_MODELS_comparison.png"
    )
    save_combined_comparison(
        cmp_png_path, img, gt_mask, all_results,
        event_name, args.threshold,
    )
    print(f"  Saved: {cmp_png_path.name}")

    # ── Metrics CSV ──────────────────────────────────────────────
    csv_path = safe_output_path(out_dir / f"cmp_{event_name}_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model", "params", "dice", "iou", "precision",
            "recall", "inference_seconds",
        ])
        w.writeheader()
        w.writerows(metrics_rows)
    print(f"  Saved: {csv_path.name}")

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Prediction complete for event: {event_name}")
    print(f"  Output directory: {out_dir.resolve()}")
    print(f"  Models evaluated: {len(all_results)}")
    if gt_mask is not None:
        print(f"\n  {'Model':<35s} {'Dice':>8s} {'IoU':>8s}")
        print(f"  {'─'*55}")
        for label, info in all_results.items():
            m = info["metrics"]
            if m:
                print(f"  {label:<35s} {m['dice']:8.4f} {m['iou']:8.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

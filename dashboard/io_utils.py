"""
io_utils.py — Discover events/models, load npy/png/csv, HDF5 + groups.txt.
Includes the critical align() helper that ensures all rasters share the same shape.
"""
from __future__ import annotations

import re
import os
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from config import (
    DEFAULT_RESULTS_DIR, MODEL_KEYS, CSV_MODEL_ALIASES,
    CANADA_H5_NAME, CANADA_GROUPS_SUFFIX, RENDER_MAX_PX,
)

# ── Pattern ────────────────────────────────────────────────────────────────────
_NPY_PATTERN = re.compile(
    r"cmp_(?P<event>.+?)_(?P<model>unet|deeplab|fcsiam|segformer|rf|calmoe)"
    r"_(?P<kind>prob|mask)\.npy$"
)
_PNG_PATTERN = re.compile(
    r"cmp_(?P<event>.+?)_(?P<model>unet|deeplab|fcsiam|segformer|rf|calmoe)"
    r"_prediction\.png$"
)
_ALL_PATTERN = re.compile(r"cmp_(?P<event>.+?)_ALL_MODELS_comparison.*\.png$")
_CSV_PATTERN = re.compile(r"cmp_(?P<event>.+?)_metrics.*\.csv$")


# ── align() — the core safety helper ─────────────────────────────────────────
def align(*arrays: Optional[np.ndarray], max_side: int = RENDER_MAX_PX) -> list:
    """
    Align multiple 2-D rasters so they all share the same (H, W), then
    optionally downsample with a common stride so results stay consistent.

    Rules:
      - None stays None.
      - If shapes differ by ≤ 4 px on each axis → crop all to min shape.
      - If shapes differ more → resize non-prob arrays to match the first
        non-None array using nearest-neighbour.
      - After shape alignment, downsample with the same factor if max_side > 0.
    """
    valid = [(i, a) for i, a in enumerate(arrays) if a is not None]
    if not valid:
        return list(arrays)

    # Target shape = shape of the first non-None array
    ref_i, ref = valid[0]
    ref_h, ref_w = ref.shape[-2], ref.shape[-1]

    aligned = list(arrays)
    for i, a in valid:
        ah, aw = a.shape[-2], a.shape[-1]
        if ah == ref_h and aw == ref_w:
            continue
        # Crop if close
        if abs(ah - ref_h) <= 4 and abs(aw - ref_w) <= 4:
            mh, mw = min(ah, ref_h), min(aw, ref_w)
            if a.ndim == 2:
                aligned[i] = a[:mh, :mw]
            else:
                aligned[i] = a[:, :mh, :mw]
            # Also crop reference if needed
            if i == ref_i or (aligned[ref_i] is not None and
                               (aligned[ref_i].shape[-2] > mh or aligned[ref_i].shape[-1] > mw)):
                r = aligned[ref_i]
                aligned[ref_i] = r[:mh, :mw] if r.ndim == 2 else r[:, :mh, :mw]
                ref_h, ref_w = mh, mw
        else:
            # Resize to ref shape using nearest-neighbour
            aligned[i] = _resize_nn(a, ref_h, ref_w)

    # Now downsample all with the same factor
    if max_side > 0:
        final_h = aligned[ref_i].shape[-2] if aligned[ref_i] is not None else ref_h
        final_w = aligned[ref_i].shape[-1] if aligned[ref_i] is not None else ref_w
        factor = max(final_h / max_side, final_w / max_side, 1.0)
        if factor > 1.0:
            sh = max(1, int(final_h / factor))
            sw = max(1, int(final_w / factor))
            for i, a in enumerate(aligned):
                if a is not None:
                    aligned[i] = _resize_nn(a, sh, sw)

    return aligned


def _resize_nn(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest-neighbour resize for 2-D or 3-D (C,H,W) arrays."""
    if arr.ndim == 2:
        im = Image.fromarray(arr if arr.dtype == np.uint8 else arr.astype(np.float32))
        im = im.resize((w, h), Image.NEAREST)
        return np.array(im)
    # (C, H, W)
    out = np.zeros((arr.shape[0], h, w), dtype=arr.dtype)
    for c in range(arr.shape[0]):
        im = Image.fromarray(arr[c] if arr[c].dtype == np.uint8 else arr[c].astype(np.float32))
        im = im.resize((w, h), Image.NEAREST)
        out[c] = np.array(im)
    return out


def maybe_downsample(arr: np.ndarray, max_px: int = RENDER_MAX_PX) -> np.ndarray:
    """Single-array convenience wrapper around align()."""
    if arr is None:
        return arr
    result = align(arr, max_side=max_px)
    return result[0]


# ── Event discovery ────────────────────────────────────────────────────────────
def discover_events(results_dir: Path) -> list[str]:
    events: set[str] = set()
    if not results_dir.is_dir():
        return []

    def _scan(folder: Path):
        for f in folder.iterdir():
            for pat in (_NPY_PATTERN, _PNG_PATTERN):
                m = pat.match(f.name)
                if m:
                    events.add(m.group("event"))
            m2 = _ALL_PATTERN.match(f.name)
            if m2:
                events.add(m2.group("event"))
            m3 = _CSV_PATTERN.match(f.name)
            if m3:
                events.add(m3.group("event"))

    _scan(results_dir)
    for child in results_dir.iterdir():
        if child.is_dir() and not child.name.startswith("."):
            _scan(child)

    return sorted(events)


def get_event_folder(results_dir: Path, event: str) -> Path:
    sub = results_dir / event
    if sub.is_dir() and any(sub.iterdir()):
        return sub
    return results_dir


def discover_models_for_event(results_dir: Path, event: str) -> list[str]:
    folder = get_event_folder(results_dir, event)
    found = []
    for key in MODEL_KEYS:
        prob_path = folder / f"cmp_{event}_{key}_prob.npy"
        png_path  = folder / f"cmp_{event}_{key}_prediction.png"
        if prob_path.exists() or png_path.exists():
            found.append(key)
    return found


# ── Cached loaders ─────────────────────────────────────────────────────────────
def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def load_prob(results_dir_str: str, event: str, model: str) -> Optional[np.ndarray]:
    results_dir = Path(results_dir_str)
    folder = get_event_folder(results_dir, event)
    path = folder / f"cmp_{event}_{model}_prob.npy"
    if not path.exists():
        return None
    try:
        return np.load(str(path), mmap_mode="r").astype(np.float32)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_mask(results_dir_str: str, event: str, model: str) -> Optional[np.ndarray]:
    results_dir = Path(results_dir_str)
    folder = get_event_folder(results_dir, event)
    mask_path = folder / f"cmp_{event}_{model}_mask.npy"
    if mask_path.exists():
        try:
            return np.load(str(mask_path), mmap_mode="r").astype(np.uint8)
        except Exception:
            pass
    return None


@st.cache_data(show_spinner=False)
def load_metrics_csv(results_dir_str: str, event: str) -> Optional[pd.DataFrame]:
    results_dir = Path(results_dir_str)
    folder = get_event_folder(results_dir, event)
    csv_path = folder / f"cmp_{event}_metrics.csv"
    if not csv_path.exists():
        candidates = sorted(folder.glob(f"cmp_{event}_metrics*.csv"), reverse=True)
        if candidates:
            csv_path = candidates[0]
        else:
            return None
    try:
        df = pd.read_csv(csv_path)
        df["model"] = df["model"].str.strip().str.lower().replace(CSV_MODEL_ALIASES)
        return df
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_prediction_png(results_dir_str: str, event: str, model: str) -> Optional[Image.Image]:
    results_dir = Path(results_dir_str)
    folder = get_event_folder(results_dir, event)
    path = folder / f"cmp_{event}_{model}_prediction.png"
    if not path.exists():
        return None
    try:
        return Image.open(path).copy()  # .copy() to avoid mmap issues
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_all_models_png(results_dir_str: str, event: str) -> Optional[Image.Image]:
    results_dir = Path(results_dir_str)
    folder = get_event_folder(results_dir, event)
    path = folder / f"cmp_{event}_ALL_MODELS_comparison.png"
    if not path.exists():
        candidates = sorted(folder.glob(f"cmp_{event}_ALL_MODELS_comparison*.png"), reverse=True)
        if candidates:
            path = candidates[0]
        else:
            return None
    try:
        return Image.open(path).copy()
    except Exception:
        return None


# ── Canada HDF5 ────────────────────────────────────────────────────────────────
def get_canada_h5_path(canada_dir: Path) -> Optional[Path]:
    p = canada_dir / CANADA_H5_NAME
    return p if p.exists() else None


def read_canada_groups(canada_dir: Path) -> list[str]:
    txt_path = canada_dir / (CANADA_H5_NAME.replace("-uint16.h5", "") + CANADA_GROUPS_SUFFIX)
    if not txt_path.exists():
        candidates = list(canada_dir.glob("*-groups.txt"))
        if not candidates:
            return []
        txt_path = candidates[0]
    events = []
    try:
        with open(txt_path) as f:
            for line in f:
                line = line.strip().strip("'()\\r\\n ,")
                if line.startswith("/20") and line.count("/") == 2:
                    parts = line.strip("/").split("/")
                    if len(parts) == 2:
                        events.append(parts[1])
    except Exception:
        pass
    return sorted(events)


@st.cache_data(show_spinner=False)
def load_event_h5(h5_path_str: str, event: str):
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from unified_dataset import load_event_from_h5
        return load_event_from_h5(h5_path_str, event)
    except Exception:
        return None, None


# ── Wildfire processed ─────────────────────────────────────────────────────────
def discover_wildfire_events(wildfire_proc_dir: Path) -> list[str]:
    if not wildfire_proc_dir.is_dir():
        return []
    return sorted(set(
        p.stem.replace("_image", "")
        for p in wildfire_proc_dir.glob("*_image.npy")
    ))


# ── RGB from 12-channel array ─────────────────────────────────────────────────
def make_rgb(img12: np.ndarray, post: bool = False) -> Optional[np.ndarray]:
    if img12 is None or img12.shape[0] < 12:
        return None
    s = 9 if post else 6
    rgb = np.stack([img12[s], img12[s + 1], img12[s + 2]], axis=-1)
    rgb = np.clip(rgb * 3.0, 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


# ── Results library scanning (for ResultsLib page) ────────────────────────────
def scan_all_results(results_dir: Path) -> pd.DataFrame:
    """
    Scan RESULTS_DIR and return a DataFrame with one row per event.
    Columns: event, year, province, has_csv, has_grid, models_with_png,
             models_with_npy, best_model, best_dice, n_models, last_modified
    """
    rows = []
    events = discover_events(results_dir)
    for ev in events:
        folder = get_event_folder(results_dir, ev)
        # Parse year/province from CA_YYYY_XX_NNN pattern
        parts = ev.split("_")
        year  = parts[1] if len(parts) > 1 and parts[1].isdigit() else ""
        prov  = parts[2] if len(parts) > 2 else ""

        has_csv  = (folder / f"cmp_{ev}_metrics.csv").exists()
        has_grid = any(folder.glob(f"cmp_{ev}_ALL_MODELS_comparison*.png"))

        png_models = [k for k in MODEL_KEYS if (folder / f"cmp_{ev}_{k}_prediction.png").exists()]
        npy_models = [k for k in MODEL_KEYS if (folder / f"cmp_{ev}_{k}_prob.npy").exists()]

        best_model, best_dice = "", 0.0
        if has_csv:
            try:
                df = pd.read_csv(folder / f"cmp_{ev}_metrics.csv")
                df["model"] = df["model"].str.strip().str.lower().replace(CSV_MODEL_ALIASES)
                df["dice"]  = pd.to_numeric(df.get("dice", 0), errors="coerce").fillna(0)
                if len(df):
                    idx = df["dice"].idxmax()
                    best_model = str(df.loc[idx, "model"])
                    best_dice  = float(df.loc[idx, "dice"])
            except Exception:
                pass

        # Last modified = most recently modified file in folder
        try:
            mtimes = [f.stat().st_mtime for f in folder.glob(f"cmp_{ev}_*") if f.is_file()]
            last_mod = max(mtimes) if mtimes else 0.0
        except Exception:
            last_mod = 0.0

        rows.append(dict(
            event=ev, year=year, province=prov,
            has_csv=has_csv, has_grid=has_grid,
            models_with_png=",".join(png_models),
            models_with_npy=",".join(npy_models),
            best_model=best_model, best_dice=best_dice,
            n_models=max(len(png_models), len(npy_models)),
            last_modified=last_mod,
        ))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def disk_size_str(path: Path) -> str:
    if not path.exists():
        return "N/A"
    if path.is_file():
        sz = path.stat().st_size
    else:
        sz = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if sz < 1024:
            return f"{sz:.1f} {unit}"
        sz /= 1024
    return f"{sz:.1f} PB"

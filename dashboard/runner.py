"""
runner.py — Subprocess prediction runner, device detection, error->hint mapping.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Generator

import streamlit as st

from config import PROJECT_ROOT

# ── Device detection ───────────────────────────────────────────────────────────
def detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return f"cuda ({torch.cuda.get_device_name(0)})"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps (Apple Silicon)"
        return "cpu"
    except ImportError:
        return "cpu (torch not installed)"


# ── Error → hint table ─────────────────────────────────────────────────────────
ERROR_HINTS = [
    ("STRICT CHANNEL MISMATCH",       "The model expects 12 channels but your input has a different number. Check --input_dir contains s1_pre/s1_post/s2_pre/s2_post with correct band counts."),
    ("FileNotFoundError",             "Input directory or HDF5 file not found. Check the path in the sidebar."),
    ("rasterio",                      "rasterio is not installed. Run: pip install rasterio"),
    ("Event .* not found in HDF5",    "The event key does not exist in the HDF5 file. Use the groups.txt browser to find the correct event name."),
    ("No module named",               "A Python dependency is missing. Run: pip install -r dashboard/requirements.txt"),
    ("RuntimeError: CUDA",            "CUDA error — switch to CPU or MPS using --device cpu in the run options."),
    ("Cannot load raster",            "File format not supported. Convert to .npy or .tif format first."),
    ("out of memory",                 "GPU/MPS out of memory. Try reducing --patch size to 128 or switching to --device cpu."),
    ("Checkpoint not found",          "Model checkpoint is missing in --ckpt-dir. Check that training completed."),
    ("RF checkpoint not found",       "Random Forest model not trained yet. Run train_compare.py or train_wildfire_compare.py first."),
]


def get_error_hint(stderr: str) -> Optional[str]:
    import re
    for pattern, hint in ERROR_HINTS:
        if re.search(pattern, stderr, re.IGNORECASE):
            return hint
    return None


# Models accepted by the CLI scripts (calmoe is always run internally)
_SCRIPT_VALID_MODELS = {"unet", "deeplab", "fcsiam", "segformer", "rf"}


# ── Script runner ──────────────────────────────────────────────────────────────
def build_predict_cmd(
    dataset: str,          # "canada" or "wildfire"
    event: str,
    weights_path: str,
    ckpt_dir: str,
    results_dir: str,
    threshold: float,
    patch_size: int,
    overlap: int,
    device: str,
    models_filter: Optional[list[str]] = None,
    input_dir: Optional[str] = None,
) -> list[str]:
    """Build the CLI command list for the correct predict script.
    NOTE: 'calmoe' is always run internally by the scripts via best_model.pth;
    it must NOT be passed to --models (not a valid CLI choice).
    """
    # Strip calmoe — it is not a valid --models choice for either script
    if models_filter:
        models_filter = [m for m in models_filter if m in _SCRIPT_VALID_MODELS]
    # Default = all 5 script models (calmoe runs automatically)
    if not models_filter:
        models_filter = sorted(_SCRIPT_VALID_MODELS)

    if dataset == "wildfire":
        script = str(PROJECT_ROOT / "predict_wildfire_compare.py")
        cmd = [
            sys.executable, script,
            "--event", event,
            "--ckpt-dir", ckpt_dir,
            "--out-dir",  results_dir,
            "--threshold", str(threshold),
            "--patch",    str(patch_size),
            "--overlap",  str(overlap),
            "--models",
        ] + models_filter
    else:
        script = str(PROJECT_ROOT / "predict_compare.py")
        cmd = [
            sys.executable, script,
            "--out-dir",   results_dir,
            "--threshold", str(threshold),
            "--patch",     str(patch_size),
            "--overlap",   str(overlap),
            "--weights",   weights_path,
            "--models",
        ] + models_filter
        if event and not input_dir:
            cmd += ["--event", event]
        if input_dir:
            cmd += ["--input_dir", input_dir]

    return cmd


def run_predict_streaming(cmd: list[str]) -> Generator[str, None, None]:
    """Run a command and yield stdout lines in real time."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    for line in proc.stdout:
        yield line
    proc.wait()
    if proc.returncode != 0:
        yield f"\n[EXIT CODE {proc.returncode}]\n"

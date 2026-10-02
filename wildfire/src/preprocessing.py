import argparse
import csv
import glob
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
PROCESSED_DIR = PROJECT_ROOT / "processed"
CSV_PATH = DATASET_DIR / "satellite_data.csv"


def parse_date(value):
    if pd.isna(value):
        return None

    raw = str(value).strip()
    if not raw:
        return None

    candidates = [
        "%d/%m/%Y",
        "%Y-%m-%d",
        "%d/%m/%Y %H:%M",
        "%Y-%m-%d %H:%M",
    ]
    for fmt in candidates:
        try:
            return datetime.strptime(raw[:10] if len(raw) >= 10 else raw, fmt)
        except ValueError:
            continue
    return None


def find_event_folder(folder_name, dataset_root=DATASET_DIR):
    matches = sorted(dataset_root.glob(f"**/{folder_name}"))
    if matches:
        return matches[0]
    return None


def extract_date_from_filename(path):
    name = Path(path).name
    match = __import__("re").search(r"(\d{4}-\d{2}-\d{2})", name)
    if not match:
        raise ValueError(f"Could not parse date from filename: {path}")
    return datetime.strptime(match.group(1), "%Y-%m-%d")


def normalize_percentile_2_98(data):
    arr = np.asarray(data, dtype=np.float32)
    out = np.zeros_like(arr, dtype=np.float32)

    for i in range(arr.shape[0]):
        band = arr[i]
        valid = np.isfinite(band)
        if not np.any(valid):
            continue
        low = float(np.percentile(band[valid], 2))
        high = float(np.percentile(band[valid], 98))
        if np.isclose(high, low):
            continue
        clipped = np.clip(band, low, high)
        out[i] = (clipped - low) / (high - low)

    return out


def load_raster_channels(path, channels_to_keep):
    with rasterio.open(path) as src:
        arr = src.read(out_dtype="float32")
    if channels_to_keep is None:
        return arr.astype(np.float32)
    return arr[:channels_to_keep].astype(np.float32)


def load_mask(path):
    with rasterio.open(path) as src:
        mask = src.read(1)
    return (mask > 0).astype(np.uint8)


def validate_processed_image(image, mask):
    if image.ndim != 3:
        raise ValueError(f"Expected 3D image, got shape {image.shape}")
    if image.shape[0] != 30:
        raise ValueError(f"Expected 30 channels, got {image.shape[0]}")
    if image.shape[1:] != mask.shape:
        raise ValueError(
            f"Image and mask shape mismatch: image={image.shape}, mask={mask.shape}"
        )
    if not np.isfinite(image).all():
        raise ValueError("Image contains NaN or Inf values")
    if image.min() < 0.0 or image.max() > 1.0 + 1e-6:
        raise ValueError(
            f"Image value range is outside [0,1]: min={image.min()}, max={image.max()}"
        )
    unique_mask = np.unique(mask)
    if not set(unique_mask.tolist()).issubset({0, 1}):
        raise ValueError(f"Mask contains unexpected classes: {unique_mask}")
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    return mask


def select_event_files(event_dir, activation_date):
    if event_dir is None:
        raise FileNotFoundError("Event folder not found")

    s1_files = sorted(glob.glob(os.path.join(str(event_dir), "sentinel1_*.tiff")))
    s2_files = sorted(glob.glob(os.path.join(str(event_dir), "sentinel2_*.tiff")))

    if not s1_files or not s2_files:
        raise FileNotFoundError(f"No Sentinel-1 or Sentinel-2 TIFFs found in {event_dir}")

    s1_before = [f for f in s1_files if extract_date_from_filename(f) < activation_date]
    s1_after = [f for f in s1_files if extract_date_from_filename(f) >= activation_date]
    s2_before = [f for f in s2_files if extract_date_from_filename(f) < activation_date]
    s2_after = [f for f in s2_files if extract_date_from_filename(f) >= activation_date]

    if not s1_before or not s1_after or not s2_before or not s2_after:
        raise ValueError(
            "Missing complete before/after pair: "
            f"S1_before={len(s1_before)}, S1_after={len(s1_after)}, "
            f"S2_before={len(s2_before)}, S2_after={len(s2_after)}"
        )

    s1_before_file = max(s1_before, key=extract_date_from_filename)
    s1_after_file = min(s1_after, key=extract_date_from_filename)
    s2_before_file = max(s2_before, key=extract_date_from_filename)
    s2_after_file = min(s2_after, key=extract_date_from_filename)

    return {
        "s1_before": s1_before_file,
        "s1_after": s1_after_file,
        "s2_before": s2_before_file,
        "s2_after": s2_after_file,
    }


def process_event(event_name, dataset_root=DATASET_DIR, processed_dir=PROCESSED_DIR):
    event_dir = find_event_folder(event_name, dataset_root)
    if event_dir is None:
        raise FileNotFoundError(f"Event folder not found for {event_name}")

    csv_df = pd.read_csv(CSV_PATH)
    row = csv_df[csv_df["folder"] == event_name]
    if row.empty:
        raise ValueError(f"No metadata row found for {event_name}")

    activation = parse_date(row.iloc[0]["activation_date"])
    if activation is None:
        raise ValueError(f"Activation date missing for {event_name}")

    files = select_event_files(event_dir, activation)

    s2_before = load_raster_channels(files["s2_before"], 12)
    s1_before = load_raster_channels(files["s1_before"], 3)
    s2_after = load_raster_channels(files["s2_after"], 12)
    s1_after = load_raster_channels(files["s1_after"], 3)

    s1_before = np.nan_to_num(s1_before, nan=0.0, posinf=0.0, neginf=0.0)
    s1_after = np.nan_to_num(s1_after, nan=0.0, posinf=0.0, neginf=0.0)

    s2_before_norm = normalize_percentile_2_98(s2_before)
    s1_before_norm = normalize_percentile_2_98(s1_before)
    s2_after_norm = normalize_percentile_2_98(s2_after)
    s1_after_norm = normalize_percentile_2_98(s1_after)

    image = np.concatenate(
        [s2_before_norm, s1_before_norm, s2_after_norm, s1_after_norm], axis=0
    ).astype(np.float32)

    mask_path = sorted(event_dir.glob("*mask*.tiff"))
    if not mask_path:
        raise FileNotFoundError(f"Mask TIFF not found for {event_name}")
    mask = load_mask(str(mask_path[0]))
    mask = validate_processed_image(image, mask.astype(np.uint8))

    processed_dir.mkdir(parents=True, exist_ok=True)
    np.save(processed_dir / f"{event_name}_image.npy", image)
    np.save(processed_dir / f"{event_name}_mask.npy", mask)

    return {
        "event": event_name,
        "image": image,
        "mask": mask,
        "activation_date": activation.strftime("%Y-%m-%d"),
        "s1_before": os.path.basename(files["s1_before"]),
        "s1_after": os.path.basename(files["s1_after"]),
        "s2_before": os.path.basename(files["s2_before"]),
        "s2_after": os.path.basename(files["s2_after"]),
        "image_shape": image.shape,
        "mask_shape": mask.shape,
        "status": "processed",
    }


def build_processing_report(report_path, rows):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "event",
        "activation_date",
        "s1_before",
        "s1_after",
        "s2_before",
        "s2_after",
        "image_shape",
        "mask_shape",
        "status",
    ]

    with open(report_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "event": row["event"],
                "activation_date": row["activation_date"],
                "s1_before": row["s1_before"],
                "s1_after": row["s1_after"],
                "s2_before": row["s2_before"],
                "s2_after": row["s2_after"],
                "image_shape": row["image_shape"],
                "mask_shape": row["mask_shape"],
                "status": row["status"],
            })


def process_all_valid_events():
    df = pd.read_csv(CSV_PATH)
    rows = []
    total = 0
    valid = 0
    skipped = 0
    errors = 0

    for _, row in df.iterrows():
        total += 1
        event_name = str(row["folder"]).strip()
        if not event_name:
            continue

        try:
            activation = parse_date(row["activation_date"])
            if activation is None:
                raise ValueError("Missing activation date")
            event_dir = find_event_folder(event_name, DATASET_DIR)
            if event_dir is None:
                raise FileNotFoundError("Folder not found")
            result = process_event(event_name)
            valid += 1
            rows.append(result)
        except Exception as exc:
            errors += 1
            skipped += 1
            rows.append({
                "event": event_name,
                "activation_date": str(row.get("activation_date", "")),
                "s1_before": "",
                "s1_after": "",
                "s2_before": "",
                "s2_after": "",
                "image_shape": "",
                "mask_shape": "",
                "status": f"skipped: {exc}",
            })

    build_processing_report(PROCESSED_DIR / "processing_report.csv", rows)
    print("Total events:", total)
    print("Valid events:", valid)
    print("Processed events:", valid)
    print("Skipped events:", skipped)
    print("Errors:", errors)


def main():
    parser = argparse.ArgumentParser(description="Fire dataset preprocessing")
    parser.add_argument("--event", type=str, default=None, help="Single event folder name to process")
    parser.add_argument("--all", action="store_true", help="Process all valid events from the dataset CSV")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if args.event:
        result = process_event(args.event)
        print("Processed event:", result["event"])
        print("Image shape:", result["image"].shape)
        print("Mask shape:", result["mask"].shape)
        print("Image dtype:", result["image"].dtype)
        print("Mask dtype:", result["mask"].dtype)
        print("Image range:", float(result["image"].min()), float(result["image"].max()))
        print("Mask values:", np.unique(result["mask"]))
        return

    if args.all:
        process_all_valid_events()
        return

    # Default: validate the known working example event.
    example_event = "EMSR226_01DABA_02GRADING_MAP_v1_vector"
    result = process_event(example_event)
    print("Demo preprocessing complete for:", example_event)
    print("Image shape:", result["image"].shape)
    print("Mask shape:", result["mask"].shape)
    print("Image dtype:", result["image"].dtype)
    print("Mask dtype:", result["mask"].dtype)
    print("Image range:", float(result["image"].min()), float(result["image"].max()))
    print("Mask values:", np.unique(result["mask"]))


if __name__ == "__main__":
    main()

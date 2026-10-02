#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.visualize import generate_event_visualization

PROJECT_ROOT = ROOT
PROCESSED_DIR = PROJECT_ROOT / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    event_name = "EMSR226_01DABA_02GRADING_MAP_v1_vector"
    image = np.load(PROCESSED_DIR / f"{event_name}_image.npy").astype(np.float32)
    gt = np.load(PROCESSED_DIR / f"{event_name}_mask.npy").astype(np.uint8)
    pred = np.load(PROCESSED_DIR / f"{event_name}_fusion_prediction.npy").astype(np.float32)
    pred = (pred >= 0.5).astype(np.uint8)

    before_image = image[:12]
    after_image = image[15:27]
    generate_event_visualization(
        event_name,
        before_image,
        after_image,
        gt,
        pred,
        RESULTS_DIR,
        experiment="fusion",
    )
    print("Saved visualization for", event_name)


if __name__ == "__main__":
    main()

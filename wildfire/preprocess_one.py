import os
from pathlib import Path

import numpy as np

from src.preprocessing import process_event


if __name__ == "__main__":
    event_name = "EMSR226_01DABA_02GRADING_MAP_v1_vector"
    result = process_event(event_name)

    print("Event:", result["event"])
    print("Activation:", result["activation_date"])
    print("S1 BEFORE:", result["s1_before"])
    print("S1 AFTER:", result["s1_after"])
    print("S2 BEFORE:", result["s2_before"])
    print("S2 AFTER:", result["s2_after"])
    print("Image shape:", result["image"].shape)
    print("Mask shape:", result["mask"].shape)
    print("Image dtype:", result["image"].dtype)
    print("Mask dtype:", result["mask"].dtype)
    print("Image range:", float(result["image"].min()), float(result["image"].max()))
    print("Mask values:", np.unique(result["mask"]).tolist())

    output_dir = Path("processed")
    output_dir.mkdir(exist_ok=True)
    print("Saved processed sample in:", output_dir)

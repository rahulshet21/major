import os
import glob
import pandas as pd
from datetime import datetime

DATASET = "dataset"

df = pd.read_csv(os.path.join(DATASET, "satellite_data.csv"))

def parse_date(x):
    if pd.isna(x):
        return None
    x = str(x).strip()
    try:
        return datetime.strptime(x[:10], "%d/%m/%Y")
    except:
        return None

valid = []
invalid = []

for _, row in df.iterrows():

    folder = str(row["folder"])
    event_dir = os.path.join(DATASET, "**", folder)

    matches = glob.glob(event_dir, recursive=True)

    if not matches:
        invalid.append((folder, "folder not found"))
        continue

    d = matches[0]

    activation = parse_date(row["activation_date"])

    if activation is None:
        invalid.append((folder, "no activation date"))
        continue

    s1_files = sorted(glob.glob(os.path.join(d, "sentinel1_*.tiff")))
    s2_files = sorted(glob.glob(os.path.join(d, "sentinel2_*.tiff")))

    def get_date(path):
        name = os.path.basename(path)
        date = name.split("_")[1].replace(".tiff", "")
        return datetime.strptime(date, "%Y-%m-%d")

    s1_before = [f for f in s1_files if get_date(f) < activation]
    s1_after  = [f for f in s1_files if get_date(f) >= activation]

    s2_before = [f for f in s2_files if get_date(f) < activation]
    s2_after  = [f for f in s2_files if get_date(f) >= activation]

    if not s1_before or not s1_after or not s2_before or not s2_after:
        invalid.append((
            folder,
            f"S1 {len(s1_before)}/{len(s1_after)}, "
            f"S2 {len(s2_before)}/{len(s2_after)}"
        ))
        continue

    # Closest acquisition to activation date on each side
    s1_pre = max(s1_before, key=get_date)
    s1_post = min(s1_after, key=get_date)

    s2_pre = max(s2_before, key=get_date)
    s2_post = min(s2_after, key=get_date)

    valid.append({
        "folder": folder,
        "activation": activation.strftime("%Y-%m-%d"),
        "S1_BEFORE": os.path.basename(s1_pre),
        "S1_AFTER": os.path.basename(s1_post),
        "S2_BEFORE": os.path.basename(s2_pre),
        "S2_AFTER": os.path.basename(s2_post),
    })

print("\n========================================")
print("VALID 30-CHANNEL EVENTS")
print("========================================")
print("Total:", len(valid))

for x in valid:
    print(
        "\n", x["folder"],
        "\n Activation:", x["activation"],
        "\n S1 BEFORE:", x["S1_BEFORE"],
        "\n S1 AFTER :", x["S1_AFTER"],
        "\n S2 BEFORE:", x["S2_BEFORE"],
        "\n S2 AFTER :", x["S2_AFTER"]
    )

print("\n========================================")
print("INVALID EVENTS")
print("========================================")
print("Total:", len(invalid))

for x in invalid:
    print(x[0], "->", x[1])

import argparse
import random
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed"
SPLITS_DIR = PROJECT_ROOT / "splits"
REPORT_PATH = PROCESSED_DIR / "processing_report.csv"


def create_event_split(seed=42):
    report = pd.read_csv(REPORT_PATH)
    valid_events = [
        str(row["event"])
        for _, row in report.iterrows()
        if str(row["status"]).startswith("processed")
    ]

    if not valid_events:
        raise ValueError("No valid events found in processing report")

    rng = random.Random(seed)
    events = valid_events[:]
    rng.shuffle(events)

    n_total = len(events)
    n_train = int(round(n_total * 0.70))
    n_val = int(round(n_total * 0.15))
    n_test = n_total - n_train - n_val

    # Keep the split event-level and deterministic.
    train_events = events[:n_train]
    val_events = events[n_train:n_train + n_val]
    test_events = events[n_train + n_val:]

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, group in [
        ("train_events.txt", train_events),
        ("val_events.txt", val_events),
        ("test_events.txt", test_events),
    ]:
        path = SPLITS_DIR / name
        with open(path, "w", encoding="utf-8") as f:
            for event in group:
                f.write(f"{event}\n")

    summary = {
        "seed": seed,
        "n_total": n_total,
        "n_train": len(train_events),
        "n_val": len(val_events),
        "n_test": len(test_events),
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Create event-level train/validation/test splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = create_event_split(seed=args.seed)
    print(summary)


if __name__ == "__main__":
    main()

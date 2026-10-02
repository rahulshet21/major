#!/usr/bin/env python3

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.split import create_event_split


if __name__ == "__main__":
    summary = create_event_split(seed=42)
    print(summary)

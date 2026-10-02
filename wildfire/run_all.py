#!/usr/bin/env python3

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd):
    print("\n>>>", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT), check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run the complete wildfire burned-area pipeline")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-visuals", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-patches-per-event", type=int, default=12)
    args = parser.parse_args()

    if not args.skip_preprocess:
        run([sys.executable, "scripts/preprocess_all.py"])
    if not args.skip_split:
        run([sys.executable, "scripts/create_event_split.py"])
    if not args.skip_train:
        for experiment in ["s1", "s2", "fusion"]:
            run([
                sys.executable,
                "scripts/train_experiment.py",
                "--experiment",
                experiment,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--max-patches-per-event",
                str(args.max_patches_per_event),
                "--patch-size",
                "256",
                "--overlap",
                "64",
            ])
    if not args.skip_eval:
        run([sys.executable, "scripts/evaluate_experiments.py"])
    if not args.skip_visuals:
        run([sys.executable, "scripts/visualize_results.py"])

    print("\nFull pipeline completed.")


if __name__ == "__main__":
    main()

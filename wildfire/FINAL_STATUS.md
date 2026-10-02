# Final project status

## Completed
- Verified 30-channel preprocessing output for the example wildfire event.
- Processed the valid event-level dataset and saved the report in `processed/processing_report.csv`.
- Created reproducible event-level splits under `splits/`.
- Built patch-based dataset extraction for 256x256 tiles with overlap.
- Implemented U-Net pipelines for S2-only, S1-only, and 30-channel fusion experiments.
- Trained and saved best checkpoints in `checkpoints/`.
- Evaluated the held-out test set and generated comparison metrics.
- Produced visual predictions and saved them under `results/`.

## Actual metrics
The test-set results from the current project run are:

| Model/Input | IoU | Dice | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Fusion | 0.516419 | 0.609553 | 0.665445 | 0.702386 | 0.609553 |
| S2-only | 0.074000 | 0.097078 | 0.140109 | 0.076617 | 0.097078 |
| S1-only | 0.001714 | 0.003416 | 0.174677 | 0.001942 | 0.003416 |

## Important note
The actual event-level valid set in this workspace was 57 events after excluding incomplete Sentinel-1 before/after acquisitions. This was determined from the generated processing report and the dataset on disk, not by assuming the brief's 59 count.

## Main entry points
- `./venv/bin/python preprocess_one.py`
- `./venv/bin/python scripts/preprocess_all.py`
- `./venv/bin/python scripts/create_event_split.py`
- `./venv/bin/python scripts/train_experiment.py --experiment fusion --epochs 20 --batch-size 4 --patch-size 256 --overlap 64`
- `./venv/bin/python scripts/evaluate_experiments.py`
- `./venv/bin/python scripts/visualize_results.py`

## Current state
The main research pipeline is working and documented. The multimodal fusion U-Net is the strongest performer on the held-out test set.

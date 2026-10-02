# Multimodal SAR–Multispectral Wildfire Burned-Area Detection

## Project goal
This project builds a deep-learning pipeline for burned-area detection using Sentinel-1 SAR and Sentinel-2 multispectral imagery with before/after event observations and a U-Net segmentation model.

## Dataset
The local dataset is already available in the `dataset/` folder. The project currently processes the valid events from the CSV metadata and excludes events that do not have a full usable before/after Sentinel-1 and Sentinel-2 pairing.

## Main workflow
1. Read event metadata from the CSV.
2. Select the nearest valid Sentinel-1 and Sentinel-2 before/after acquisition for each event.
3. Keep only the required channels (S2: first 12, S1: first 3).
4. Apply 2nd/98th percentile normalization and clip to `[0, 1]`.
5. Build a 30-channel fusion image as:
   - S2 before (12)
   - S1 before (3)
   - S2 after (12)
   - S1 after (3)
6. Convert the mask to binary burned/not burned using `(mask > 0)`.
7. Create event-level train/validation/test splits.
8. Extract 256x256 patches with overlap.
9. Train U-Net models for the three experiments:
   - S2-only
   - S1-only
   - 30-channel fusion
10. Evaluate on the held-out event-level test set.

## Repository structure
- `src/preprocessing.py` — dataset preprocessing and event selection
- `src/split.py` — reproducible event-level splits
- `src/dataset.py` — patch extraction dataset
- `src/model.py` — U-Net builder
- `src/losses.py` — BCE + Dice loss
- `src/train.py` — training loop and checkpointing
- `src/inference.py` — sliding-window prediction
- `scripts/train_experiment.py` — command-line training entry point

## Environment setup
```bash
cd wildfire
./venv/bin/python -m pip install -r requirements.txt
```

## Example preprocessing verification
```bash
cd wildfire
./venv/bin/python preprocess_one.py
```
Expected output includes:
- image shape `(30, H, W)`
- mask shape `(H, W)`
- image dtype `float32`
- mask dtype `uint8`
- image range `[0, 1]`
- mask values `[0, 1]`

## Full preprocessing run
```bash
cd wildfire
./venv/bin/python scripts/preprocess_all.py
```

## Create event-level split
```bash
cd wildfire
./venv/bin/python scripts/create_event_split.py
```

## Train an experiment
```bash
cd wildfire
./venv/bin/python scripts/train_experiment.py --experiment fusion --epochs 5 --batch-size 4 --max-patches-per-event 12 --patch-size 256 --overlap 64
```
You can replace `fusion` with `s1` or `s2`.

## Run the complete pipeline
```bash
cd wildfire
./venv/bin/python run_all.py --epochs 5 --batch-size 4 --max-patches-per-event 12
```
This runs preprocessing, split creation, training for all three experiments, evaluation, and visualization.

## Inference
```bash
cd wildfire
./venv/bin/python src/inference.py --event EMSR226_01DABA_02GRADING_MAP_v1_vector --experiment fusion
```

## Actual measured results
The current validated test-set comparison is:

| Model/Input | IoU | Dice | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Fusion | 0.516419 | 0.609553 | 0.665445 | 0.702386 | 0.609553 |
| S2-only | 0.074000 | 0.097078 | 0.140109 | 0.076617 | 0.097078 |
| S1-only | 0.001714 | 0.003416 | 0.174677 | 0.001942 | 0.003416 |

## Current status
The research pipeline has been completed and validated for the required three U-Net experiments. The code is structured for CPU, MPS, and CUDA devices, with automatic hardware detection.

## Known limitations
- Some events are missing valid before/after Sentinel-1 pairs and are excluded.
- The binary mask assumption uses all non-zero labels as burned fire pixels.
- Model performance depends on patch size, class imbalance, and the current dataset coverage.
- The current implementation is a reproducible baseline rather than a final tuned production model.

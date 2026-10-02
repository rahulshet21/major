# Wildfire Burned-Area Mapping Usage Guide

This project segments wildfire burned areas using Sentinel-1 (radar) and Sentinel-2 (optical) satellite imagery (pre-fire and post-fire).
The core model architecture is **CA-LMoETransUNet** (CNN encoder $\rightarrow$ Transformer + Mixture of Experts bottleneck $\rightarrow$ CNN decoder).

Input Tensors: **12 channels** (`float32` in `[0.0, 1.0]` range):
- Channels `0..2`  : Sentinel-1 Pre-fire (`ND`, `VH`, `VV`)
- Channels `3..5`  : Sentinel-1 Post-fire (`ND`, `VH`, `VV`)
- Channels `6..8`  : Sentinel-2 Pre-fire (`B4`, `B8`, `B12`)
- Channels `9..11` : Sentinel-2 Post-fire (`B4`, `B8`, `B12`)

---

## Hardware Acceleration (macOS)
Device fallback is automatically applied across all scripts:
$$\text{CUDA (GPU)} \longrightarrow \text{MPS (Apple Silicon GPU)} \longrightarrow \text{CPU}$$

---

## 1. Inspecting the HDF5 File
Inspect the dataset structure (years, events, dataset keys, shapes, dtypes, value ranges, and CRS):

```bash
# Print HDF5 structure summary file
cat h5_structure.txt

# Inspect a specific event in the HDF5 dataset via predict_folder.py
python3 predict_folder.py --h5 candata/wildfire-s1s2-alos-dataset-canada-uint16.h5 --event CA_2017_BC_1157 --inspect
```

---

## 2. Extracting Patches (Pre-caching)
Extract 256x256 image patches into memory-mapped `.npy` files for instant training:

```bash
python3 preextract_patches.py \
  --h5 candata/wildfire-s1s2-alos-dataset-canada-uint16.h5 \
  --groups-txt candata/wildfire-s1s2-alos-dataset-canada-groups.txt \
  --patches 16 \
  --patch-size 256 \
  --out patches_ca.npz
```

---

## 3. Training the CA-LMoETransUNet Model
Train the model using the unified dataset loader (automatically uses `.npy` memmap if present, legacy 30ch `.npy`, or direct HDF5):

```bash
python3 train_ca_lmoe.py \
  --h5 candata/wildfire-s1s2-alos-dataset-canada-uint16.h5 \
  --epochs 20 \
  --batch-size 4 \
  --patch-size 256 \
  --lr 1e-4 \
  --out-dir checkpoints_ca
```

---

## 4. Predicting from a Folder of Sentinel-1 / Sentinel-2 Rasters
Predict burned areas from a directory containing GeoTIFF or `.npy` files for S1 and S2 (pre/post):

```bash
# Auto-discover S1/S2 pre/post rasters, tile, predict, and export probability .npy,
# georeferenced mask GeoTIFF (if CRS exists), and a 4-panel visual PNG:
python3 predict_folder.py \
  --input_dir test_event_folder \
  --weights best_model.pth \
  --out_dir results/ \
  --threshold 0.5 \
  --patch 256 \
  --overlap 32

# Inspect folder rasters without running prediction:
python3 predict_folder.py --input_dir test_event_folder --inspect
```

---

## 5. Predicting from an HDF5 Event
Run inference directly on an event stored inside the Canada HDF5 dataset:

```bash
python3 predict_folder.py \
  --h5 candata/wildfire-s1s2-alos-dataset-canada-uint16.h5 \
  --event CA_2017_BC_1157 \
  --weights best_model.pth \
  --out_dir results/ \
  --threshold 0.5
```

Alternatively, use `predict_ca_lmoe.py`:

```bash
python3 predict_ca_lmoe.py \
  --h5 candata/wildfire-s1s2-alos-dataset-canada-uint16.h5 \
  --event CA_2017_BC_1157 \
  --model best_model.pth \
  --out-dir results/
```

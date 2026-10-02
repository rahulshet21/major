# Wildfire Burned Area Mapping Project Overview

This document provides a comprehensive, start-to-finish explanation of your wildfire mapping project. It details the data used, the neural network architecture, and the step-by-step workflow for training and inference across your environments (`candata` and Colab).

---

## 1. Project Goal
The objective is to train a deep learning model to automatically detect and map **burned areas** caused by wildfires using satellite imagery. 
Instead of relying solely on optical images (which can be blocked by clouds or smoke), the project fuses **Sentinel-1 (radar)** and **Sentinel-2 (optical)** satellite data from both *before* and *after* a wildfire event.

## 2. The Dataset (Canada HDF5)
**File:** `wildfire-s1s2-alos-dataset-canada-uint16.h5` (21 GB)
This is a massive, highly structured dataset containing historical wildfire events in Canada (grouped by years like 2017, 2018, etc.).

For each wildfire event (e.g., `CA_2017_BC_1157`), the dataset contains:
- **Sentinel-1 (S1) Radar Data:** 3 bands (e.g., VV, VH) from both pre-fire and post-fire dates. (6 channels total).
- **Sentinel-2 (S2) Optical Data:** 3 bands (e.g., Red, NIR, SWIR) from both pre-fire and post-fire dates. (6 channels total).
- **Mask (Ground Truth):** A binary polygon mask indicating exactly where the fire burned.

**Total Input Channels:** 12 channels per pixel (6 from S1 + 6 from S2).

---

## 3. The Model Architecture: `CA-LMoETransUNet`
Your model is a **Channel Attention Local Mixture of Experts Transformer U-Net**. It is a highly advanced architecture for semantic segmentation.

Here is how it processes the data:
1. **CNN Encoder (U-Net part):** The 12-channel image is passed through a series of `ResidualBlocks` (Conv2D -> BatchNorm -> ReLU) which gradually downsample the image while extracting deep spatial features.
2. **Transformer Bottleneck:** At the deepest layer, the spatial features are flattened into tokens and passed into a Transformer block. 
3. **Mixture of Experts (MoE):** Instead of a standard Transformer attention mechanism, the model uses MoE. It has 3 "Expert" networks. A gating mechanism decides which expert is best suited to process which part of the image, allowing the model to specialize in different types of terrain or fire patterns.
4. **CNN Decoder:** The tokens are reshaped back into a grid and passed up through `ConvTranspose2d` layers. Skip connections from the encoder are added back in to retain high-resolution spatial details.
5. **Output Head:** A final convolution outputs a 2-channel probability map (Fire vs. No-Fire) for every single pixel.

---

## 4. The End-to-End Workflow & Scripts

### Phase 1: Data Pre-Extraction (`preextract_patches.py`)
- **The Problem:** The HDF5 dataset is 21GB. Training requires randomly cropping 256x256 "patches" from different events thousands of times per epoch. On a CPU or standard hard drive, reading random chunks from a massive compressed HDF5 file causes extreme I/O lag, making training impossible.
- **The Solution:** This script scans the HDF5 file once, extracts 5,248 random 256x256 patches, and saves them into flat, uncompressed `.npy` (numpy) files (`patches_images.npy` and `patches_masks.npy`). 
- **Result:** These `.npy` files can be "Memory-Mapped" (`mmap_mode='r'`), meaning the OS reads them instantly from the disk without loading the whole 16GB into RAM.

### Phase 2: Local Training Setup (`train_ca_lmoe.py`)
- This is the main training script on your Mac.
- It defines the `MemmapDataset` to load the pre-extracted patches efficiently.
- It calculates the **Dice Loss** (which measures overlap between predicted fire and actual fire) combined with **Cross-Entropy Loss**.
- **The Bottleneck:** Training 3.78 million parameters on 5,248 patches takes roughly ~25 minutes *per epoch* on a Mac CPU. 20 epochs would take ~8 hours.

### Phase 3: Cloud Training (`CA_LMoETransUNet_Colab.ipynb`)
- **The Solution:** We moved the training to Google Colab to access a free NVIDIA T4 GPU.
- **The RAM Crash Fix:** We discovered that loading the 16.5GB of `.npy` arrays directly into Colab's 12.7GB RAM limit crashed the server. 
- **The Final Colab Workflow:** The notebook was rewritten so that you upload a compressed `.npz` file to Google Drive. The notebook copies it to the Colab local disk, unzips it, and memory-maps it. 
- **Result:** Training time dropped from 8 hours (CPU) to **15 minutes** (GPU). The script outputs `best_model.pth`.

### Phase 4: Inference and Visualization (`predict_ca_lmoe.py`)
- Once `best_model.pth` is downloaded back to your Mac, this script is used to visualize the results.
- **How it works:** It loads an entire wildfire event from the HDF5 file. Because an event is much larger than 256x256, the script automatically breaks the massive image into a grid of 256x256 patches, runs the model on each patch, and seamlessly stitches them back together into a massive prediction mask.
- **Output:** It generates a side-by-side PNG image (`prediction_CA_2017_BC_1157.png`) showing the RGB image, the Ground Truth, the Model's Probability Heatmap, and a final Red Overlay of the detected fire.

---

## 5. Summary of Folders
- **`candata/`**: Contains the core logic, datasets, and scripts specifically written and adapted to process the 12-channel Canada dataset.
- **`wildfire/`**: (Your earlier folder or Colab Drive folder). In the context of Google Drive, this is where the Colab notebook reads the `.npz` data from and saves the `checkpoints_ca` (including the model weights and training curves).

## 6. Next Steps for Analysis
Now that you have a working pipeline, to analyze and improve the project you can:
1. **Run `predict_ca_lmoe.py` on different events** (e.g. `--event CA_2017_BC_1005`) to see if the model struggles with specific topographies or cloud cover.
2. **Tweak the Model:** Increase `embed_dim` to 256 or `num_experts` to 4 in Colab to see if a heavier model gets a higher Validation Dice Score (currently ~0.69).
3. **Thresholding:** The inference script currently uses `> 0.5` probability to declare a fire. You can analyze the heatmaps to see if raising this to `0.7` reduces false positives.

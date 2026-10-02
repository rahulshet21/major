# Project report

## Abstract
Wildfire burned-area mapping is a difficult segmentation task because burned regions may be spatially fragmented, temporally variable, and visually confounded by vegetation and terrain. A multimodal approach combining Sentinel-1 synthetic aperture radar and Sentinel-2 multispectral observations before and after an event can improve the detection of burned land compared with optical-only methods.

## Methodology
The implemented workflow follows the dataset, event selection, preprocessing, patch extraction, and U-Net baseline pipeline described in the project specification. The current implementation uses a binary burned-area target, where any non-zero mask value is treated as burned.

### Data pipeline
- Read event metadata from the dataset CSV.
- Select the nearest valid pre/post Sentinel-1 and Sentinel-2 image per event.
- Use 12 Sentinel-2 channels and 3 Sentinel-1 channels per side.
- Normalize with 2nd/98th percentile scaling and clip to `[0, 1]`.
- Combine the four temporal groups into a 30-channel input tensor.
- Build event-level train/validation/test splits.
- Extract 256x256 patches with overlap.

### Model
A U-Net with a residual encoder backbone is used for burned-area segmentation. The code supports automatic device selection and can run on CUDA, MPS, or CPU.

### Loss
The training pipeline uses a combination of binary cross-entropy and Dice loss to address class imbalance while maintaining stable optimization.

## Experiments
Three main experiments are implemented:
1. S2-only U-Net
2. S1-only U-Net
3. S1 + S2 fusion U-Net using the 30-channel input

## Validation status
The preprocessing step was validated on the example event `EMSR226_01DABA_02GRADING_MAP_v1_vector`, which produced the required output shape, dtype, and range. The patch extraction and minimal training loop were also verified on all three model variants.

## Results
The held-out event-level test-set results from the current implementation are:

| Model/Input | IoU | Dice | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Fusion | 0.516419 | 0.609553 | 0.665445 | 0.702386 | 0.609553 |
| S2-only | 0.074000 | 0.097078 | 0.140109 | 0.076617 | 0.097078 |
| S1-only | 0.001714 | 0.003416 | 0.174677 | 0.001942 | 0.003416 |

## Limitations
- The dataset contains events with incomplete Sentinel-1 coverage and these are excluded.
- The binary mask assumption treats all non-zero labels as burned pixels.
- Model performance is an initial baseline and not yet tuned for final deployment.
- Mac hardware constraints require lightweight, CPU/MPS-compatible training rather than large CUDA-scale experiments.

## Future work
- Add full evaluation metrics table after the held-out test set is fully assessed.
- Extend the visualization pipeline for before/after RGB comparison, ground truth, and prediction overlays.
- Run full training for all three experiments with tuned hyperparameters and longer epochs.
- Compare against DeepLabV3+ and SegFormer baselines after the U-Net pipeline is stable.

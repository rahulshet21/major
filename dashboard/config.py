"""
config.py — Paths, model registry, constants.
"""
from __future__ import annotations
from pathlib import Path

# Auto-detect PROJECT_ROOT as parent of the dashboard/ folder
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Default paths (relative to PROJECT_ROOT)
DEFAULT_RESULTS_DIR   = PROJECT_ROOT / "results"
DEFAULT_CANADA_DIR    = PROJECT_ROOT / "candata"
DEFAULT_WILDFIRE_DIR  = PROJECT_ROOT / "wildfire" / "dataset"
DEFAULT_WILDFIRE_PROC = PROJECT_ROOT / "wildfire" / "processed"
BEST_MODEL_PATH       = PROJECT_ROOT / "best_model.pth"
CANADA_H5_NAME        = "wildfire-s1s2-alos-dataset-canada-uint16.h5"
CANADA_GROUPS_SUFFIX  = "-groups.txt"
UPLOADS_DIR           = Path(__file__).parent / "uploads"

# Model registry
MODELS = [
    dict(
        key="unet",
        label="U-Net (ResNet-34)",
        color="#4fc3f7",
        blurb="Encoder–decoder with skip connections. Backbone: ResNet-34. 24.5M params. Strong general-purpose segmenter.",
        architecture="smp.Unet(ResNet-34)",
    ),
    dict(
        key="deeplab",
        label="DeepLabV3+",
        color="#a5d6a7",
        blurb="Atrous Spatial Pyramid Pooling + encoder–decoder. Backbone: ResNet-50. 26.8M params. Excels at multi-scale context.",
        architecture="smp.DeepLabV3Plus(ResNet-50)",
    ),
    dict(
        key="fcsiam",
        label="FC-Siam",
        color="#ffcc80",
        blurb="Siamese fully-convolutional change-detection network. 7.7M params. Directly compares pre/post feature maps.",
        architecture="FC-Siamese encoder-decoder",
    ),
    dict(
        key="segformer",
        label="SegFormer (MiT-B0)",
        color="#ce93d8",
        blurb="Hierarchical transformer encoder + lightweight MLP decoder. 3.8M params. Efficient and accurate.",
        architecture="smp.Segformer(MiT-B0)",
    ),
    dict(
        key="rf",
        label="Random Forest",
        color="#ef9a9a",
        blurb="Non-deep baseline. 20 spectral + radar indices (dNBR, dVH, dVV…). No GPU required.",
        architecture="sklearn.RandomForestClassifier(n_estimators=200)",
    ),
    dict(
        key="calmoe",
        label="CA-LMoETransUNet",
        color="#ff6b35",
        blurb="Existing model. 3.78M params. Channel Attention + Mixture-of-Experts Transformer bottleneck. Dice 0.6924 / IoU 0.5337 on validation.",
        architecture="12→64→128→256→128→MoE(3 experts, 4 heads)→decoder→2-class head",
    ),
]

MODEL_KEYS   = [m["key"]   for m in MODELS]
MODEL_LABELS = {m["key"]: m["label"] for m in MODELS}
MODEL_COLORS = {m["key"]: m["color"] for m in MODELS}

# CSV alias normalisation: map "calmoe (existing)" → "calmoe"
CSV_MODEL_ALIASES: dict[str, str] = {
    "calmoe (existing)": "calmoe",
    "ca-lmoetransunet (existing)": "calmoe",
}

# 12-channel layout (Canada dataset)
CH12_TABLE = [
    ("S1 Pre – ND",   "Radar",   "Normalised difference (VH−VV)/(VH+VV)"),
    ("S1 Pre – VH",   "Radar",   "Cross-polarisation backscatter (pre-fire)"),
    ("S1 Pre – VV",   "Radar",   "Co-polarisation backscatter (pre-fire)"),
    ("S1 Post – ND",  "Radar",   "Normalised difference (post-fire)"),
    ("S1 Post – VH",  "Radar",   "Cross-polarisation backscatter (post-fire)"),
    ("S1 Post – VV",  "Radar",   "Co-polarisation backscatter (post-fire)"),
    ("S2 Pre – B4",   "Optical", "Red band (pre-fire)"),
    ("S2 Pre – B8",   "Optical", "Near-Infrared band (pre-fire)"),
    ("S2 Pre – B12",  "Optical", "Short-Wave Infrared band (pre-fire)"),
    ("S2 Post – B4",  "Optical", "Red band (post-fire)"),
    ("S2 Post – B8",  "Optical", "Near-Infrared band (post-fire)"),
    ("S2 Post – B12", "Optical", "Short-Wave Infrared band (post-fire)"),
]

# Physics of fire signatures
PHYSICS_TABLE = [
    ("NIR (B8)",  "↓ 65 %", "Canopy destroyed → loss of green vegetation reflectance"),
    ("SWIR (B12)","↑ sharp","Ash + char reflect strongly in SWIR"),
    ("Red (B4)",  "↑ mild", "Bare soil exposed"),
    ("VH SAR",    "↓ 55 %", "Canopy volume removed → loss of volume scattering"),
    ("VV SAR",    "slight ↓","Surface scattering from soil remains"),
    ("dNBR",      "↑ large","(NIR−SWIR)/(NIR+SWIR) drops post-fire → dNBR = NBR_pre − NBR_post"),
]

# Pixel size for area calculation (metres)
DEFAULT_PIXEL_SIZE_M = 10.0

# Downsampling cap for rendering
RENDER_MAX_PX = 1600

# PR/ROC curve stratified sample cap
CURVE_SAMPLE_SIZE = 200_000

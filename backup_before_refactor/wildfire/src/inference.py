import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model import build_unet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sliding_window_predict(model, image, patch_size=256, overlap=64, device=None):
    if device is None:
        device = get_device()

    h, w = image.shape[1], image.shape[2]
    stride = patch_size - overlap
    result = np.zeros((1, h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)

    for y0 in range(0, h, stride):
        y1 = min(y0 + patch_size, h)
        for x0 in range(0, w, stride):
            x1 = min(x0 + patch_size, w)
            patch = image[:, y0:y1, x0:x1]
            if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
                pad_h = patch_size - patch.shape[1]
                pad_w = patch_size - patch.shape[2]
                patch = np.pad(patch, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            patch_t = torch.from_numpy(patch).unsqueeze(0).float().to(device)
            with torch.no_grad():
                logits = model(patch_t)
                prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
            prob = prob[: y1 - y0, : x1 - x0]
            result[0, y0:y1, x0:x1] += prob
            weight_map[y0:y1, x0:x1] += 1.0

    return result[0] / np.maximum(weight_map, 1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run wildfire burned-area inference")
    parser.add_argument("--event", type=str, default="EMSR226_01DABA_02GRADING_MAP_v1_vector")
    parser.add_argument("--experiment", choices=["s2", "s1", "fusion"], default="fusion")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    args = parser.parse_args()

    device = get_device()
    event_name = args.event
    input_path = PROCESSED_DIR / f"{event_name}_image.npy"
    image = np.load(input_path).astype(np.float32)

    if args.experiment == "s2":
        image = image[:12]
    elif args.experiment == "s1":
        image = image[12:15]

    model = build_unet(in_channels=image.shape[0])
    checkpoint = CHECKPOINT_DIR / f"{args.experiment}_best.pth"
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    pred = sliding_window_predict(model, image, patch_size=args.patch_size, overlap=args.overlap, device=device)
    np.save(PROCESSED_DIR / f"{event_name}_{args.experiment}_prediction.npy", pred)
    print("Saved prediction for", event_name, "shape=", pred.shape)

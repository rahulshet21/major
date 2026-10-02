import os
import argparse
import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
from train_ca_lmoe import CALMoETransUNet  # Import the model definition

def get_event_data(h5_path, event_name):
    """Load the full event data from the HDF5 file."""
    with h5py.File(h5_path, 'r') as f:
        year = event_name.split('_')[1]
        
        if year not in f or event_name not in f[year]:
            raise ValueError(f"Event {event_name} not found in {h5_path} under year {year}")
        
        g = f[year][event_name]
        
        # Load S1 (3 bands * 2 times) and S2 (3 bands * 2 times) for pre and post
        s1_pre  = g['s1'][0, :, :, :].astype(np.float32)
        s1_post = g['s1'][1, :, :, :].astype(np.float32)
        s2_pre  = g['s2'][0, :, :, :].astype(np.float32)
        s2_post = g['s2'][1, :, :, :].astype(np.float32)
        
        # Ground truth mask (poly is band 0)
        mask = g['mask'][0, :, :].astype(np.int64)
        
        # Combine channels (12 total)
        img = np.concatenate([s1_pre, s1_post, s2_pre, s2_post], axis=0)
        
        # Normalize to [0,1]
        img = np.clip(img / 65535.0, 0, 1)
        mask = (mask > 0).astype(np.int64)
        
        return img, mask

def predict_full_image(model, img, device, patch_size=256, batch_size=8):
    """Run model on full image by breaking it into patches."""
    C, H, W = img.shape
    
    # Pad image to be multiple of patch_size
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0,0), (0,pad_h), (0,pad_w)), mode='reflect')
    
    _, pad_H, pad_W = img.shape
    
    pred_mask = np.zeros((pad_H, pad_W), dtype=np.float32)
    
    patches = []
    coords = []
    
    for y in range(0, pad_H, patch_size):
        for x in range(0, pad_W, patch_size):
            patch = img[:, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            coords.append((y, x))
            
    # Process in batches
    for i in range(0, len(patches), batch_size):
        batch = np.stack(patches[i:i+batch_size])
        batch_tensor = torch.from_numpy(batch).to(device)
        
        # Handle NaNs just like in training
        batch_tensor = torch.nan_to_num(batch_tensor, nan=0.0)
        
        with torch.no_grad():
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=1)[:, 1] # Probability of fire (class 1)
            probs = probs.cpu().numpy()
            
        for j, (y, x) in enumerate(coords[i:i+batch_size]):
            pred_mask[y:y+patch_size, x:x+patch_size] = probs[j]
            
    # Remove padding
    pred_mask = pred_mask[:H, :W]
    return pred_mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", default="wildfire-s1s2-alos-dataset-canada-uint16.h5")
    parser.add_argument("--model", required=True, help="Path to best_model.pth")
    parser.add_argument("--event", default="CA_2017_BC_1157", help="Event name to predict")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading model...")
    model = CALMoETransUNet(in_channels=12, out_channels=2, embed_dim=128, num_experts=3, transformer_depth=1)
    
    checkpoint = torch.load(args.model, map_location='cpu')
    # If the model was saved with DataParallel, keys might have 'module.' prefix.
    # The colab notebook saved it directly.
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded. Validation Dice was: {checkpoint.get('val_dice', 0):.4f}")
    
    print(f"\nLoading event {args.event} from HDF5...")
    img, mask_gt = get_event_data(args.h5, args.event)
    
    print("Running prediction (this may take a minute on CPU)...")
    pred_prob = predict_full_image(model, img, device)
    
    # Threshold probabilities to get binary mask
    pred_binary = (pred_prob > 0.5).astype(np.uint8)
    
    print("Saving visualization...")
    # Use SWIR (band 11, index 9 in S2), NIR (band 8, index 6), Red (band 4, index 2) for background if possible
    # We will just use an RGB representation from the 12 channels (assuming last 10 are S2)
    # S2 indices in the 12-channel stack: 2 to 11.
    # Red = index 4 (S2 band 4), Green = index 3 (S2 band 3), Blue = index 2 (S2 band 2)
    r = img[4]
    g = img[3]
    b = img[2]
    rgb = np.stack([r, g, b], axis=-1)
    # Brighten for visibility
    rgb = np.clip(rgb * 3.0, 0, 1)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(rgb)
    axes[0].set_title(f"Post-Fire RGB\n{args.event}")
    axes[0].axis('off')
    
    axes[1].imshow(mask_gt, cmap='gray')
    axes[1].set_title("Ground Truth (Poly)")
    axes[1].axis('off')
    
    axes[2].imshow(pred_prob, cmap='inferno')
    axes[2].set_title("Predicted Probability")
    axes[2].axis('off')
    
    axes[3].imshow(rgb)
    # Overlay predictions in red
    axes[3].imshow(np.where(pred_binary==1, 1.0, np.nan), cmap='Reds', alpha=0.6, vmin=0, vmax=1)
    axes[3].set_title("Overlay (Prediction in Red)")
    axes[3].axis('off')
    
    plt.tight_layout()
    out_file = f"prediction_{args.event}.png"
    plt.savefig(out_file, dpi=150)
    print(f"\nDone! Result saved to {out_file}")

if __name__ == "__main__":
    main()

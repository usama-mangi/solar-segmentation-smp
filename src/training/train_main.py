import os
import json
import random
import argparse
from glob import glob
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms # Added import

# Project imports
from ..models.factory import build_model
from .loss import FocalLoss

# ====== Dataset (Corrected) ======
class SegTileDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=256):
        self.images = image_paths
        self.masks = mask_paths
        self.img_size = img_size
        
        # Define standard ImageNet normalization
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Open images
        img = Image.open(self.images[idx]).convert("RGB")
        msk = Image.open(self.masks[idx]).convert("L")
        
        # Resize
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        msk = msk.resize((self.img_size, self.img_size), Image.NEAREST)

        # Apply Normalization (Replaces manual division and transpose)
        # ToTensor converts [0, 255] -> [0.0, 1.0] and moves channels to first dim (C, H, W)
        img = self.normalize(img)
        
        # Process Mask
        msk = np.array(msk).astype(np.float32) / 255.0
        msk = (msk > 0.5).astype(np.float32)
        msk = np.expand_dims(msk, 0) # (1, H, W)

        return img, torch.from_numpy(msk)

# ====== Metrics ======
def dice_coef(pred, target, eps=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + eps) / (pred.sum() + target.sum() + eps)

# ====== Training Loop ======
def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    it = 0
    for imgs, masks in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        logits = model(imgs)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            d = dice_coef(probs, masks).item()
        total_dice += d
        it += 1
    if it == 0: return 0.0, 0.0
    return total_loss/it, total_dice/it

def valid_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    it = 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            d = dice_coef(probs, masks).item()
            total_dice += d
            it += 1
    if it==0: return 0.0, 0.0
    return total_loss/it, total_dice/it

def load_simclr_weights(model, encoder_weights_path, device):
    if not os.path.exists(encoder_weights_path):
        print(f"Warning: SSL weights not found at {encoder_weights_path}. Using ImageNet.")
        return

    print(f"Loading SSL weights from {encoder_weights_path}...")
    sd = torch.load(encoder_weights_path, map_location=device, weights_only=False)
    
    if 'state_dict' in sd:
        sd = sd['state_dict']
    
    new_sd = {}
    for k, v in sd.items():
        # Clean common SSL prefixes to match SMP encoder keys
        k_clean = k.replace("module.", "").replace("backbone.", "").replace("encoder.", "")
        new_sd[k_clean] = v
        
    try:
        # Load strictly into the encoder part of the SMP model
        msg = model.encoder.load_state_dict(new_sd, strict=False)
        print(f"Weights loaded. Missing keys: {len(msg.missing_keys)}")
    except Exception as e:
        print(f"Error loading weights: {e}")

def train_segmentation(
    data_dir, 
    save_dir, 
    arch="unet", 
    backbone="resnet50", 
    encoder_weights_path=None, 
    epochs=20, 
    batch_size=8
):
    LR = 3e-5
    IMG_SIZE = 256
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Data Setup
    images_root_dir = os.path.join(data_dir, "ORIGINAL")
    masks_root_dir = os.path.join(data_dir, "annotated_images")
    all_imgs = sorted(glob(os.path.join(images_root_dir, "*")))
    all_msks = sorted(glob(os.path.join(masks_root_dir, "*")))
    
    if not all_imgs:
        print(f"Error: No images found in {images_root_dir}")
        return

    # Split
    n = len(all_imgs)
    ntr = int(0.8 * n)
    indices = list(range(n))
    random.seed(42) 
    random.shuffle(indices)
    
    train_ds = SegTileDataset([all_imgs[i] for i in indices[:ntr]], [all_msks[i] for i in indices[:ntr]], img_size=IMG_SIZE)
    val_ds = SegTileDataset([all_imgs[i] for i in indices[ntr:]], [all_msks[i] for i in indices[ntr:]], img_size=IMG_SIZE)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # 2. Build Model
    print(f"Building {arch} with {backbone}...")
    model = build_model(arch, backbone, n_classes=1, encoder_weights="imagenet").to(DEVICE)
    
    # 3. Load SimCLR Weights (if provided)
    if encoder_weights_path:
        load_simclr_weights(model, encoder_weights_path, DEVICE)

    # 4. Setup Training
    loss_fn = FocalLoss(alpha=0.4, gamma=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    best_dice = 0.0
    history = {"train_loss": [], "val_dice": []}

    print("\n=== Starting Training (Fine-tuning) ===")
    for epoch in range(epochs):
        tr_loss, tr_dice = train_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        val_loss, val_dice = valid_epoch(model, val_loader, loss_fn, DEVICE)
        
        history["train_loss"].append(tr_loss)
        history["val_dice"].append(val_dice)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {tr_loss:.4f} | Val Dice: {val_dice:.4f}")
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(history, f)
    
    return best_dice

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--arch", default="unet")
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--weights", default=None)
    args = parser.parse_args()
    
    train_segmentation(args.data_dir, args.save_dir, args.arch, args.backbone, args.weights)
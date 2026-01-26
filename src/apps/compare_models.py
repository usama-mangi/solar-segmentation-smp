import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from torchvision import transforms
import segmentation_models_pytorch as smp

# --- Configuration ---
# Set this to True to generate difference maps (Red=False Positive, Blue=False Negative)
SHOW_ERROR_MAPS = True 
IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. Model Builder (Matches your Factory logic) ---
def load_model(arch, backbone, weights_path, n_classes=1):
    print(f"Loading {arch} + {backbone} from {weights_path}...")
    
    # Factory Logic
    archs = {
        "unet": smp.Unet,
        "fpn": smp.FPN,
        "pspnet": smp.PSPNet,
        "deeplabv3": smp.DeepLabV3,
    }
    
    # Initialize model structure
    # Note: We use encoder_weights=None because we will load our own custom weights
    model = archs[arch](
        encoder_name=backbone, 
        encoder_weights=None, 
        in_channels=3, 
        classes=n_classes
    )
    
    # Load Checkpoint
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=True)
        
        # Unpack if saved inside 'state_dict' key
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            
        # Clean keys if necessary (e.g., if saved with DDP 'module.' prefix)
        clean_sd = {}
        for k, v in state_dict.items():
            new_k = k.replace("module.", "") 
            clean_sd[new_k] = v
            
        model.load_state_dict(clean_sd)
    else:
        print(f"ERROR: Weight file not found at {weights_path}")
        return None

    model.to(DEVICE)
    model.eval()
    return model

# --- 2. Inference Helpers ---
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(model, img_pil):
    """Runs inference on a single PIL image."""
    img_tensor = preprocess(img_pil).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)
        mask = (probs > 0.5).float().cpu().numpy()[0, 0] # shape (H, W)
        
    return mask

def compute_dice(pred, gt):
    """Computes Dice score for a single pair of masks."""
    intersection = np.sum(pred * gt)
    return (2. * intersection) / (np.sum(pred) + np.sum(gt) + 1e-6)

# --- 3. Visualization Core ---
def make_error_map(pred, gt):
    """
    Creates an RGB image showing errors.
    Green = True Positive
    Red = False Positive (Model predicted solar, but it wasn't)
    Blue = False Negative (Model missed the solar panel)
    Black = True Negative
    """
    # Initialize RGB canvas
    h, w = pred.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # True Positive (Green)
    tp = np.logical_and(pred == 1, gt == 1)
    rgb[tp] = [0, 255, 0]
    
    # False Positive (Red)
    fp = np.logical_and(pred == 1, gt == 0)
    rgb[fp] = [255, 0, 0]
    
    # False Negative (Blue)
    fn = np.logical_and(pred == 0, gt == 1)
    rgb[fn] = [0, 0, 255]
    
    return rgb

def visualize_comparison(data_dir, output_dir, models_config):
    """
    Main driver function.
    models_config: List of dicts [{'name': 'SimCLR', 'arch': 'unet', ...}, ...]
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Setup Models
    loaded_models = []
    for cfg in models_config:
        m = load_model(cfg['arch'], cfg['backbone'], cfg['path'])
        if m:
            loaded_models.append({'model': m, 'name': cfg['name']})
            
    if not loaded_models:
        print("No models loaded. Exiting.")
        return

    # 2. Get Data
    img_dir = os.path.join(data_dir, "ORIGINAL")
    mask_dir = os.path.join(data_dir, "annotated_images")
    img_paths = sorted(glob(os.path.join(img_dir, "*.jpg")) + glob(os.path.join(img_dir, "*.png")))[:10] # Limit to 10 for demo
    
    print(f"Found {len(img_paths)} images to visualize.")

    # 3. Run Pipeline
    for img_path in img_paths:
        fname = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, fname)
        
        if not os.path.exists(mask_path):
            continue
            
        # Load Raw Data
        original_pil = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        gt_pil = Image.open(mask_path).convert("L").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        gt_arr = (np.array(gt_pil) > 127).astype(np.float32)

        # Plot Setup
        num_models = len(loaded_models)
        rows = 2 if SHOW_ERROR_MAPS else 1
        cols = 2 + num_models # Original + GT + Models
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if rows == 1: axes = axes[None, :] # Ensure 2D array access
        
        # -- Col 1: Original --
        axes[0, 0].imshow(original_pil)
        axes[0, 0].set_title("Original")
        axes[0, 0].axis("off")
        if SHOW_ERROR_MAPS: axes[1, 0].axis("off") # Empty slot

        # -- Col 2: Ground Truth --
        axes[0, 1].imshow(gt_arr, cmap='gray')
        axes[0, 1].set_title("Ground Truth")
        axes[0, 1].axis("off")
        if SHOW_ERROR_MAPS: axes[1, 1].axis("off")

        # -- Models --
        for idx, item in enumerate(loaded_models):
            col_idx = 2 + idx
            model_name = item['name']
            
            # Predict
            pred_mask = predict_image(item['model'], original_pil)
            dice = compute_dice(pred_mask, gt_arr)
            
            # Row 1: Prediction Mask
            axes[0, col_idx].imshow(pred_mask, cmap='gray')
            axes[0, col_idx].set_title(f"{model_name}\nDice: {dice:.3f}")
            axes[0, col_idx].axis("off")
            
            # Row 2: Error Map
            if SHOW_ERROR_MAPS:
                err_map = make_error_map(pred_mask, gt_arr)
                axes[1, col_idx].imshow(err_map)
                axes[1, col_idx].set_title("Errors (R:FP, B:FN, G:TP)")
                axes[1, col_idx].axis("off")

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"compare_{fname}")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved visualization: {save_path}")

if __name__ == "__main__":
    # --- USER CONFIGURATION START ---
    
    DATA_DIRECTORY = "data/processed" # Point this to your data
    OUTPUT_DIRECTORY = "experiments/visualizations"
    
    # Define the models you want to compare here
    # Since you added an 'encoder' parameter, your saved weights likely reflect that structure.
    MODELS_TO_COMPARE = [
        {
            "name": "FPN+SimCLR+ResNet50",
            "arch": "fpn",
            "backbone": "resnet50",
            "path": "experiments/fpn_simclr_resnet50/best_model.pth" 
        },
        {
            "name": "PSPNet+SimCLR+ResNet50",
            "arch": "pspnet",
            "backbone": "resnet50",
            "path": "experiments/pspnet_simclr_resnet50/best_model.pth"
        },
        {
            "name": "UNet+SimCLR+ResNet50",
            "arch": "unet",
            "backbone": "resnet50",
            "path": "experiments/unet_simclr_resnet50/best_model.pth"
        },
        {
            "name": "FPN+MoCo+ResNet50",
            "arch": "fpn",
            "backbone": "resnet50",
            "path": "experiments/fpn_moco_resnet50/best_model.pth" 
        },
        {
            "name": "PSPNet+MoCo+ResNet50",
            "arch": "pspnet",
            "backbone": "resnet50",
            "path": "experiments/pspnet_moco_resnet50/best_model.pth"
        },
        {
            "name": "UNet+MoCo+ResNet50",
            "arch": "unet",
            "backbone": "resnet50",
            "path": "experiments/unet_moco_resnet50/best_model.pth"
        },
    ]
    # --- USER CONFIGURATION END ---

    visualize_comparison(DATA_DIRECTORY, OUTPUT_DIRECTORY, MODELS_TO_COMPARE)
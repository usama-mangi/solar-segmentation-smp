import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from torchvision import transforms
import segmentation_models_pytorch as smp

# Config
IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(arch, backbone, weights_path):
    print(f"Loading {arch} from {weights_path}...")
    
    # Define Arch (Must match your training)
    if arch == "unet":
        model = smp.Unet(encoder_name=backbone, encoder_weights=None, in_channels=3, classes=1)
    
    # Load Weights
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=DEVICE)
        # Handle 'state_dict' wrapper if present
        if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
        # Remove 'module.' prefix if DDP was used
        clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_sd)
    else:
        print("Weights not found!")
        return None

    model.to(DEVICE)
    model.eval()
    return model

def predict(model, img_path):
    # Preprocess
    preprocess = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    original = Image.open(img_path).convert("RGB")
    input_tensor = preprocess(original).unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)
        mask = (probs > 0.5).float().cpu().numpy()[0, 0]
        
    return original, mask

def run_blind_test(test_dir, output_dir, model_config):
    os.makedirs(output_dir, exist_ok=True)
    model = load_model(model_config['arch'], model_config['backbone'], model_config['path'])
    
    # Find all images
    images = sorted(glob(os.path.join(test_dir, "*.jpg")) + glob(os.path.join(test_dir, "*.png")))
    print(f"Found {len(images)} images in {test_dir}")

    for img_path in images:
        fname = os.path.basename(img_path)
        original, mask = predict(model, img_path)
        
        # Visualize Side-by-Side
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        
        ax[0].imshow(original)
        ax[0].set_title("Test Image")
        ax[0].axis("off")
        
        ax[1].imshow(mask, cmap='gray')
        ax[1].set_title("Model Prediction")
        ax[1].axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"pred_{fname}"))
        plt.close()
        print(f"Processed {fname}")

if __name__ == "__main__":
    # Point this to your unannotated folder
    TEST_DIR = "data/processed/test_images" 
    OUTPUT_DIR = "experiments/blind_predictions"
    
    # Use your best model
    MODEL_CFG = {
        "arch": "unet",
        "backbone": "resnet50",
        "path": "experiments/unet_simclr_resnet50/best_model.pth"
    }
    
    run_blind_test(TEST_DIR, OUTPUT_DIR, MODEL_CFG)
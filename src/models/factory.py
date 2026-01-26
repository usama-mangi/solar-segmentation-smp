import segmentation_models_pytorch as smp

def build_model(arch, backbone, n_classes=1, encoder_weights="imagenet"):
    """
    Factory function to build segmentation models.
    
    Args:
        arch (str): Architecture name ('unet', 'fpn', 'pspnet').
        backbone (str): Encoder name (e.g., 'resnet50', 'vgg16', 'mit_b3').
        n_classes (int): Number of output classes.
        encoder_weights (str): Pretrained weights ('imagenet' or None).
    """
    archs = {
        "unet": smp.Unet,
        "fpn": smp.FPN,
        "pspnet": smp.PSPNet,
        # Add others if needed: 'deeplabv3', 'linknet', etc.
    }
    
    if arch not in archs:
        raise ValueError(f"Architecture {arch} not supported. Choose from {list(archs.keys())}")

    # Create model with ImageNet initialization
    model = archs[arch](
        encoder_name=backbone, 
        encoder_weights=encoder_weights, 
        in_channels=3, 
        classes=n_classes
    )
    
    return model
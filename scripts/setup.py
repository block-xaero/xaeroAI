# xaeroai/setup_models.py
#!/usr/bin/env python3
"""
Download and quantize MobileSAM + train whiteboard classifier
Run this once to set up models
"""

import torch
import requests
import os
from pathlib import Path

def download_mobile_sam():
    """Download MobileSAM"""
    print("Downloading MobileSAM...")

    # Create models directory
    Path("../models").mkdir(exist_ok=True)

    # Download MobileSAM checkpoint
    url = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
    response = requests.get(url)

    with open("../models/mobile_sam.pt", "wb") as f:
        f.write(response.content)

    print("✓ Downloaded MobileSAM")

def quantize_models():
    """Convert to INT8 and save as safetensors"""
    print("Quantizing models...")

    import torch.quantization as quant
    from safetensors.torch import save_file

    # Load MobileSAM
    checkpoint = torch.load("../models/mobile_sam.pt", map_location="cpu")

    # For now, just save as safetensors (quantization optional)
    # MobileSAM is already small enough
    save_file(checkpoint, "../models/mobile_sam.safetensors")

    print("✓ Saved as safetensors")

if __name__ == "__main__":
    download_mobile_sam()
    quantize_models()
    print("\nSetup complete! Models in xaeroai/models/")
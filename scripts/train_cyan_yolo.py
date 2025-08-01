#!/usr/bin/env python3
"""
Train YOLOv8 for Cyan digitization
"""
from ultralytics import YOLO

def train_cyan_model():
    """Train YOLOv8 on Cyan creative content"""
    
    # Load pretrained nano model
    model = YOLO('models/yolo/pretrained/yolov8n.pt')
    
    # Train on Cyan data
    results = model.train(
        data='models/yolo/xaero_digitization.yaml',
        epochs=100,
        batch=16,
        device='mps',  # Use Apple Silicon GPU
        project='models/yolo/custom',
        name='cyan_digitization',
        
        # Optimization for creative content
        lr0=0.01,        # Initial learning rate
        patience=20,     # Early stopping patience
        save_period=10,  # Save checkpoint every 10 epochs
        
        # Data augmentation for sketches
        degrees=15,      # Rotation augmentation
        translate=0.1,   # Translation augmentation
        scale=0.2,       # Scale augmentation
        fliplr=0.5,      # Horizontal flip
    )
    
    print("🎉 Training complete!")
    print(f"📊 Results saved to: {results.save_dir}")
    
    # Export trained model
    model.export(format='onnx', optimize=True)
    print("📦 Model exported to ONNX format")

if __name__ == "__main__":
    train_cyan_model()

#!/usr/bin/env python3
"""
Download and setup YOLOv8 models for Cyan digitization
"""
import os
from pathlib import Path
from ultralytics import YOLO
import torch

def setup_directories():
    """Create directory structure for YOLOv8 models"""
    dirs = [
        "models/yolo",
        "models/yolo/pretrained",
        "models/yolo/custom",
        "models/yolo/exports"
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created {dir_path}")

def download_yolo_models():
    """Download YOLOv8 models in different sizes"""

    models = {
        'yolov8n.pt': 'Nano (fastest, smallest)',
        'yolov8s.pt': 'Small (balanced)',
        'yolov8m.pt': 'Medium (more accurate)',
    }

    print("⬇️ Downloading YOLOv8 models...")
    downloaded_models = {}

    for model_name, description in models.items():
        try:
            print(f"\n🔄 Downloading {model_name} - {description}")

            # This automatically downloads the model
            model = YOLO(model_name)

            # Get model info
            model_size = model.model_info()

            # Save to our directory
            save_path = f"models/yolo/pretrained/{model_name}"

            # Copy from cache to our structure
            import shutil
            cache_path = Path.home() / ".cache" / "ultralytics" / model_name
            if cache_path.exists():
                shutil.copy(cache_path, save_path)
                file_size = os.path.getsize(save_path) / 1024 / 1024
                print(f"✅ {model_name} saved - {file_size:.1f}MB")

                downloaded_models[model_name] = {
                    'path': save_path,
                    'size_mb': file_size,
                    'description': description
                }
            else:
                print(f"⚠️ Could not find cached model for {model_name}")

        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")

    return downloaded_models

def test_yolo_models(downloaded_models):
    """Test downloaded models with a simple inference"""
    print("\n🧪 Testing YOLOv8 models...")

    # Create a dummy image for testing
    import numpy as np
    from PIL import Image

    # Create a simple test image (random colors)
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    test_image_pil = Image.fromarray(test_image)
    test_image_path = "models/yolo/test_image.jpg"
    test_image_pil.save(test_image_path)

    for model_name, info in downloaded_models.items():
        try:
            print(f"\n🔬 Testing {model_name}...")

            # Load model
            model = YOLO(info['path'])

            # Run inference
            results = model(test_image_path, verbose=False)

            # Get timing info
            print(f"✅ {model_name} inference successful")
            print(f"   Detected {len(results[0].boxes) if results[0].boxes is not None else 0} objects")

        except Exception as e:
            print(f"❌ {model_name} test failed: {e}")

    # Clean up test image
    if os.path.exists(test_image_path):
        os.remove(test_image_path)

def create_custom_training_config():
    """Create configuration files for custom training"""

    # Cyan digitization dataset config
    cyan_config = """
# Cyan Digitization Dataset Configuration
path: data/cyan_sketches  # Dataset root dir
train: images/train  # Train images (relative to 'path')
val: images/val      # Val images (relative to 'path')

# Creative content classes
names:
  0: rectangle
  1: circle  
  2: line
  3: arrow
  4: text_region
  5: garment_shape
  6: architectural_element
  7: diagram_node
  8: hand_drawn_curve
  9: annotation
"""

    config_path = "models/yolo/xaero_digitization.yaml"
    with open(config_path, 'w') as f:
        f.write(cyan_config.strip())

    print(f"📄 Created training config: {config_path}")

    # Training script template
    training_script = '''#!/usr/bin/env python3
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
'''

    script_path = "scripts/train_cyan_yolo.py"
    with open(script_path, 'w') as f:
        f.write(training_script)

    os.chmod(script_path, 0o755)  # Make executable
    print(f"🐍 Created training script: {script_path}")

def create_inference_example():
    """Create example inference code for Cyan integration"""

    inference_code = '''#!/usr/bin/env python3
"""
YOLOv8 inference for Cyan digitization
"""
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

class CyanDigitizer:
    def __init__(self, model_path="models/yolo/custom/cyan_digitization/weights/best.pt"):
        """Initialize Cyan digitization model"""
        self.model = YOLO(model_path)
        
        # Class names for creative content
        self.class_names = {
            0: 'rectangle', 1: 'circle', 2: 'line', 3: 'arrow',
            4: 'text_region', 5: 'garment_shape', 6: 'architectural_element',
            7: 'diagram_node', 8: 'hand_drawn_curve', 9: 'annotation'
        }
    
    def digitize_sketch(self, image_path):
        """Digitize a hand-drawn sketch"""
        
        # Run inference
        results = self.model(image_path)
        
        # Extract detected elements
        elements = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get class and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.class_names.get(class_id, 'unknown')
                    
                    elements.append({
                        'type': class_name,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence
                    })
        
        return elements
    
    def visualize_detection(self, image_path, output_path=None):
        """Visualize detection results"""
        results = self.model(image_path)
        
        # Plot results
        for result in results:
            plotted = result.plot()
            
            if output_path:
                cv2.imwrite(output_path, plotted)
                print(f"🖼️ Visualization saved to: {output_path}")
            else:
                # Display image
                cv2.imshow('Cyan Digitization', plotted)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    digitizer = CyanDigitizer()
    
    # Example: Process a fashion sketch
    sketch_path = "examples/fashion_sketch.jpg"
    if os.path.exists(sketch_path):
        elements = digitizer.digitize_sketch(sketch_path)
        
        print("🎨 Detected elements:")
        for element in elements:
            print(f"  {element['type']}: {element['confidence']:.2f} confidence")
        
        # Visualize
        digitizer.visualize_detection(sketch_path, "output/detected_elements.jpg")
    else:
        print(f"⚠️ Example image not found: {sketch_path}")
        print("Add your own sketch images to test!")
'''

    script_path = "scripts/cyan_inference.py"
    with open(script_path, 'w') as f:
        f.write(inference_code)

    os.chmod(script_path, 0o755)
    print(f"🔍 Created inference script: {script_path}")

def main():
    """Download and setup YOLOv8 for Cyan"""

    print("🚀 Setting up YOLOv8 for Cyan Digitization")
    print("=" * 50)

    # Install ultralytics if not present
    try:
        import ultralytics
        print("✅ Ultralytics already installed")
    except ImportError:
        print("📦 Installing ultralytics...")
        os.system("pip install ultralytics")

    # Setup directories
    setup_directories()
    print()

    # Download models
    downloaded_models = download_yolo_models()
    print()

    # Test models
    if downloaded_models:
        test_yolo_models(downloaded_models)
        print()

    # Create configuration files
    create_custom_training_config()
    print()

    # Create inference examples
    create_inference_example()
    print()

    # Summary
    print("🎉 YOLOv8 setup complete!")
    print()
    print("📁 Downloaded models:")
    for model_name, info in downloaded_models.items():
        print(f"   {model_name}: {info['size_mb']:.1f}MB - {info['description']}")

    print()
    print("🚀 Next steps:")
    print("   1. Collect Cyan sketch training data")
    print("   2. Annotate images with creative elements")
    print("   3. Run: python scripts/train_cyan_yolo.py")
    print("   4. Test with: python scripts/cyan_inference.py")
    print()
    print("💡 Recommended for Cyan:")
    print("   - Use yolov8n.pt (nano) for mobile/fast inference")
    print("   - Train on diverse creative content (fashion, architecture, diagrams)")
    print("   - Export to ONNX for cross-platform deployment")

if __name__ == "__main__":
    main()
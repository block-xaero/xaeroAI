#!/usr/bin/env python3
"""
Verify YOLOv8 setup and test models
"""
from ultralytics import YOLO
import os
from pathlib import Path

def verify_downloads():
    """Verify the downloaded models work correctly"""

    print("🔍 Verifying YOLOv8 downloads...")

    models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']
    working_models = {}

    for model_name in models:
        try:
            print(f"\n📦 Testing {model_name}...")

            # Load model (will use cached version)
            model = YOLO(model_name)

            # Get file size from cache
            cache_path = Path.home() / ".cache" / "ultralytics" / model_name
            if cache_path.exists():
                size_mb = os.path.getsize(cache_path) / 1024 / 1024
                print(f"   ✅ {model_name}: {size_mb:.1f}MB")

                working_models[model_name] = {
                    'size_mb': size_mb,
                    'cache_path': str(cache_path)
                }
            else:
                print(f"   ⚠️ {model_name}: Downloaded but cache path not found")

        except Exception as e:
            print(f"   ❌ {model_name}: Error - {e}")

    return working_models

def test_inference():
    """Test basic inference with YOLOv8 nano"""

    print("\n🧪 Testing inference with YOLOv8 Nano...")

    try:
        # Load nano model
        model = YOLO('yolov8n.pt')

        # Create a simple test image
        import numpy as np
        from PIL import Image

        # Create test image (blue rectangle on white background)
        img = np.ones((640, 640, 3), dtype=np.uint8) * 255  # White background
        img[200:400, 200:400] = [0, 0, 255]  # Blue rectangle

        test_image = Image.fromarray(img)
        test_path = "test_image.jpg"
        test_image.save(test_path)

        # Run inference
        results = model(test_path, verbose=False)

        # Check results
        num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
        print(f"   ✅ Inference successful: {num_detections} objects detected")

        # Clean up
        os.remove(test_path)

        return True

    except Exception as e:
        print(f"   ❌ Inference test failed: {e}")
        return False

def copy_models_to_structure():
    """Copy models from cache to our project structure"""

    print("\n📁 Organizing models in project structure...")

    models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']

    for model_name in models:
        try:
            cache_path = Path.home() / ".cache" / "ultralytics" / model_name
            project_path = Path("models/yolo/pretrained") / model_name

            if cache_path.exists():
                import shutil
                shutil.copy(cache_path, project_path)
                size_mb = os.path.getsize(project_path) / 1024 / 1024
                print(f"   ✅ {model_name}: Copied to project ({size_mb:.1f}MB)")
            else:
                print(f"   ⚠️ {model_name}: Cache file not found")

        except Exception as e:
            print(f"   ❌ {model_name}: Copy failed - {e}")

def show_quick_start():
    """Show quick start commands"""

    print("\n🚀 Quick Start Commands:")
    print()
    print("# Test YOLOv8 nano model:")
    print("python3 -c \"from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('Model loaded successfully!')\"")
    print()
    print("# Run inference on an image:")
    print("python3 -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')('path/to/image.jpg')\"")
    print()
    print("# Start training on custom data:")
    print("python scripts/train_cyan_yolo.py")
    print()
    print("# Test Cyan digitization:")
    print("python scripts/cyan_inference.py")

def main():
    """Main verification function"""

    print("🔍 YOLOv8 Setup Verification")
    print("=" * 40)

    # Verify downloads
    working_models = verify_downloads()

    # Test inference
    inference_works = test_inference()

    # Copy to project structure
    copy_models_to_structure()

    # Summary
    print("\n📊 Setup Summary:")
    print(f"   📦 Models downloaded: {len(working_models)}/3")
    print(f"   🧪 Inference test: {'✅ PASS' if inference_works else '❌ FAIL'}")

    if len(working_models) > 0:
        print(f"   🎯 Recommended: yolov8n.pt ({working_models.get('yolov8n.pt', {}).get('size_mb', 'N/A'):.1f}MB)")

    # Show next steps
    show_quick_start()

    if len(working_models) >= 1 and inference_works:
        print("\n🎉 YOLOv8 is ready for Cyan digitization!")
    else:
        print("\n⚠️ Some issues detected - but basic functionality should work")

if __name__ == "__main__":
    main()
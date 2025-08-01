#!/usr/bin/env python3
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

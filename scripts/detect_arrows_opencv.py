import cv2
import numpy as np
import sys
import json

def detect_arrows_and_lines(image_path, min_length=30):
    """Detect arrows and lines using OpenCV"""
    
    img = cv2.imread(image_path)
    if img is None:
        return []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Adaptive threshold for whiteboard (dark ink on light background)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(
        thresh, 
        rho=1, 
        theta=np.pi/180, 
        threshold=50,
        minLineLength=min_length,
        maxLineGap=10
    )
    
    detections = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate line length
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length < min_length:
                continue
            
            # Check for arrowhead at endpoints
            has_arrow = detect_arrowhead(thresh, x1, y1, x2, y2)
            
            # Bounding box
            min_x, max_x = min(x1, x2), max(x1, x2)
            min_y, max_y = min(y1, y2), max(y1, y2)
            
            # Add padding for thin lines
            pad = 5
            min_x = max(0, min_x - pad)
            min_y = max(0, min_y - pad)
            max_x = min(w, max_x + pad)
            max_y = min(h, max_y + pad)
            
            class_name = "solid_arrow" if has_arrow else "line"
            
            detections.append({
                "class_name": class_name,
                "confidence": 0.8,  # Fixed confidence for OpenCV detections
                "x": float(min_x),
                "y": float(min_y),
                "width": float(max_x - min_x),
                "height": float(max_y - min_y),
                "endpoints": [[int(x1), int(y1)], [int(x2), int(y2)]]
            })
    
    # Merge overlapping detections
    detections = merge_overlapping(detections)
    
    return detections


def detect_arrowhead(thresh, x1, y1, x2, y2, search_radius=20):
    """Check if there's an arrowhead at either endpoint"""
    h, w = thresh.shape
    
    for ex, ey in [(x1, y1), (x2, y2)]:
        # Extract region around endpoint
        rx1 = max(0, ex - search_radius)
        ry1 = max(0, ey - search_radius)
        rx2 = min(w, ex + search_radius)
        ry2 = min(h, ey + search_radius)
        
        region = thresh[ry1:ry2, rx1:rx2]
        if region.size == 0:
            continue
        
        # Find contours in region
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            # Approximate contour
            epsilon = 0.04 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # Arrowhead typically has 3-5 vertices (triangle-ish)
            if 3 <= len(approx) <= 7:
                area = cv2.contourArea(cnt)
                if 50 < area < 2000:  # Reasonable arrowhead size
                    return True
    
    return False


def merge_overlapping(detections, iou_thresh=0.5):
    """Merge overlapping line detections"""
    if not detections:
        return []
    
    # Sort by confidence
    detections = sorted(detections, key=lambda x: -x["confidence"])
    
    keep = []
    used = set()
    
    for i, det in enumerate(detections):
        if i in used:
            continue
        keep.append(det)
        
        for j, other in enumerate(detections[i+1:], i+1):
            if j in used:
                continue
            if compute_iou(det, other) > iou_thresh:
                used.add(j)
    
    return keep


def compute_iou(a, b):
    x1 = max(a["x"], b["x"])
    y1 = max(a["y"], b["y"])
    x2 = min(a["x"] + a["width"], b["x"] + b["width"])
    y2 = min(a["y"] + a["height"], b["y"] + b["height"])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = a["width"] * a["height"]
    area_b = b["width"] * b["height"]
    union = area_a + area_b - intersection
    
    return intersection / union if union > 0 else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_arrows_opencv.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    detections = detect_arrows_and_lines(image_path)
    
    print(f"Found {len(detections)} arrows/lines:")
    for det in detections:
        print(f"  {det['class_name']}: ({det['x']:.0f}, {det['y']:.0f}) {det['width']:.0f}x{det['height']:.0f}")
    
    # Output JSON for pipeline integration
    print("\nJSON output:")
    print(json.dumps(detections, indent=2))

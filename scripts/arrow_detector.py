"""
OpenCV-based arrow/connector detection for whiteboard images.
Used as fallback when YOLO misses thin arrows.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

def detect_connectors(
    img: np.ndarray, 
    shapes: List[Dict], 
    min_length: int = 50
) -> List[Dict]:
    """
    Detect arrows/lines connecting shapes using OpenCV.
    
    Args:
        img: BGR image (numpy array)
        shapes: List of detected shapes from YOLO, each with:
            - class_name: str
            - x, y, width, height: float (pixel coordinates)
        min_length: Minimum line length in pixels
    
    Returns:
        List of detected connectors, each with:
            - class_name: "solid_arrow"
            - confidence: float
            - x, y, width, height: bounding box
            - connects: [shape_idx_start, shape_idx_end]
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Filter to shapes that can be connected (not text labels, not arrows)
    connector_classes = ["text_label", "solid_arrow", "dashed_arrow", "line", 
                         "curved_arrow", "bidirectional_arrow"]
    shape_boxes = [s for s in shapes if s["class_name"] not in connector_classes]
    
    if len(shape_boxes) < 2:
        return []
    
    # Create mask excluding shape interiors
    mask = np.ones((h, w), dtype=np.uint8) * 255
    for box in shape_boxes:
        x, y = int(box["x"]), int(box["y"])
        bw, bh = int(box["width"]), int(box["height"])
        shrink = 10
        x, y = max(0, x + shrink), max(0, y + shrink)
        bw, bh = max(1, bw - 2*shrink), max(1, bh - 2*shrink)
        mask[y:y+bh, x:x+bw] = 0
    
    # Edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    edges = cv2.bitwise_and(edges, mask)
    
    # Detect lines
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=30,
        minLineLength=min_length,
        maxLineGap=20
    )
    
    if lines is None:
        return []
    
    # Dynamic max distance based on image size
    max_dist = max(200, int(min(h, w) * 0.1))
    
    connectors = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        start_shape = _find_nearest_shape((x1, y1), shape_boxes, max_dist)
        end_shape = _find_nearest_shape((x2, y2), shape_boxes, max_dist)
        
        if start_shape is not None and end_shape is not None and start_shape != end_shape:
            connectors.append({
                "class_name": "solid_arrow",
                "confidence": 0.7,
                "x": float(min(x1, x2) - 10),
                "y": float(min(y1, y2) - 10),
                "width": float(abs(x2 - x1) + 20),
                "height": float(abs(y2 - y1) + 20),
                "connects": [start_shape, end_shape],
            })
    
    # Deduplicate (same shape pair)
    unique = {}
    for c in connectors:
        key = tuple(sorted(c["connects"]))
        if key not in unique:
            unique[key] = c
    
    return list(unique.values())


def _find_nearest_shape(
    point: Tuple[int, int], 
    shapes: List[Dict], 
    max_dist: int
) -> Optional[int]:
    """Find index of nearest shape to a point."""
    px, py = point
    best_idx, best_dist = None, max_dist
    
    for i, s in enumerate(shapes):
        dist = _point_to_box_dist(px, py, s)
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    
    return best_idx


def _point_to_box_dist(px: int, py: int, box: Dict) -> float:
    """Distance from point to nearest edge of box."""
    bx, by = box["x"], box["y"]
    bw, bh = box["width"], box["height"]
    
    cx = max(bx, min(px, bx + bw))
    cy = max(by, min(py, by + bh))
    
    return np.sqrt((px - cx)**2 + (py - cy)**2)


if __name__ == "__main__":
    import json
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python arrow_detector.py <image_path> <shapes_json>")
        sys.exit(1)
    
    img = cv2.imread(sys.argv[1])
    shapes = json.loads(sys.argv[2])
    
    connectors = detect_connectors(img, shapes)
    print(json.dumps(connectors))

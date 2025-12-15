import cv2
import numpy as np
import random
import os
from pathlib import Path

# Classes that need more data (from training results)
WEAK_CLASSES = {
    0: "rectangle",
    4: "diamond",
    5: "triangle",
    12: "solid_arrow",
    15: "line",
    23: "dashed_line",
    24: "dotted_line",
    29: "curved_line",
}

def draw_rectangle(img, x, y, w, h):
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 2)
    return 0, x + w//2, y + h//2, w, h

def draw_diamond(img, cx, cy, size):
    pts = np.array([
        [cx, cy - size],
        [cx + size, cy],
        [cx, cy + size],
        [cx - size, cy]
    ], np.int32)
    cv2.polylines(img, [pts], True, (0, 0, 0), 2)
    return 4, cx, cy, size * 2, size * 2

def draw_triangle(img, cx, cy, size):
    pts = np.array([
        [cx, cy - size],
        [cx - size, cy + size],
        [cx + size, cy + size]
    ], np.int32)
    cv2.polylines(img, [pts], True, (0, 0, 0), 2)
    return 5, cx, cy, size * 2, size * 2

def draw_solid_arrow(img, x1, y1, x2, y2):
    cv2.arrowedLine(img, (x1, y1), (x2, y2), (0, 0, 0), 2, tipLength=0.3)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    w, h = abs(x2 - x1) + 20, abs(y2 - y1) + 20
    return 12, cx, cy, max(w, 30), max(h, 30)

def draw_dashed_line(img, x1, y1, x2, y2):
    dash_len = 10
    dx = x2 - x1
    dy = y2 - y1
    dist = int(np.sqrt(dx*dx + dy*dy))
    for i in range(0, dist, dash_len * 2):
        start = (x1 + int(dx * i / dist), y1 + int(dy * i / dist))
        end_i = min(i + dash_len, dist)
        end = (x1 + int(dx * end_i / dist), y1 + int(dy * end_i / dist))
        cv2.line(img, start, end, (0, 0, 0), 2)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    w, h = abs(x2 - x1) + 10, abs(y2 - y1) + 10
    return 23, cx, cy, max(w, 20), max(h, 20)

def generate_image(img_size=640, num_shapes=5):
    # White/off-white background with slight texture
    bg_color = random.randint(240, 255)
    img = np.full((img_size, img_size, 3), bg_color, dtype=np.uint8)

    # Add slight noise for realism
    noise = np.random.randint(-5, 5, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    labels = []
    shape_funcs = [
        lambda: draw_rectangle(img,
                               random.randint(50, img_size-150),
                               random.randint(50, img_size-150),
                               random.randint(60, 150),
                               random.randint(40, 100)),
        lambda: draw_diamond(img,
                             random.randint(100, img_size-100),
                             random.randint(100, img_size-100),
                             random.randint(30, 60)),
        lambda: draw_triangle(img,
                              random.randint(100, img_size-100),
                              random.randint(100, img_size-100),
                              random.randint(30, 60)),
        lambda: draw_solid_arrow(img,
                                 random.randint(50, img_size//2),
                                 random.randint(50, img_size-50),
                                 random.randint(img_size//2, img_size-50),
                                 random.randint(50, img_size-50)),
        lambda: draw_dashed_line(img,
                                 random.randint(50, img_size//2),
                                 random.randint(50, img_size-50),
                                 random.randint(img_size//2, img_size-50),
                                 random.randint(50, img_size-50)),
    ]

    for _ in range(num_shapes):
        func = random.choice(shape_funcs)
        try:
            class_id, cx, cy, w, h = func()
            # Convert to YOLO format (normalized)
            labels.append(f"{class_id} {cx/img_size:.6f} {cy/img_size:.6f} {w/img_size:.6f} {h/img_size:.6f}")
        except:
            pass

    return img, labels

def main():
    output_dir = Path("synthetic_data")
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)

    num_images = 200  # Generate 200 synthetic images

    for i in range(num_images):
        img, labels = generate_image(num_shapes=random.randint(3, 8))

        img_path = output_dir / "images" / "train" / f"synthetic_{i:04d}.jpg"
        label_path = output_dir / "labels" / "train" / f"synthetic_{i:04d}.txt"

        cv2.imwrite(str(img_path), img)
        with open(label_path, "w") as f:
            f.write("\n".join(labels))

        if i % 50 == 0:
            print(f"Generated {i}/{num_images}")

    print(f"Done! Generated {num_images} images in {output_dir}")
    print("Merge with your existing dataset and retrain.")

if __name__ == "__main__":
    main()

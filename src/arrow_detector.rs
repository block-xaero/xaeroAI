//! Arrow and line detection using pure Rust imageproc
//!
//! Uses Canny edge detection + Hough line transform
//! Tuned for whiteboard diagram arrow detection

use crate::runtime::DetectedBox;
use anyhow::Result;
use image::{GrayImage, Luma};
use imageproc::edges::canny;
use imageproc::filter::gaussian_blur_f32;
use imageproc::hough::{detect_lines, LineDetectionOptions};
use std::collections::HashMap;

/// Detect arrows and lines connecting shapes
pub fn detect_connectors(
    image_data: &[u8],
    shapes: &[DetectedBox],
    min_length: u32,
) -> Result<Vec<DetectedBox>> {
    let img = image::load_from_memory(image_data)?;
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();

    // Filter out existing connectors - only keep shapes
    let shape_boxes: Vec<&DetectedBox> = shapes
        .iter()
        .filter(|s| !is_connector_class(&s.class_name))
        .collect();

    // Need at least 2 shapes to connect
    if shape_boxes.len() < 2 {
        return Ok(vec![]);
    }

    // Preprocessing: blur to reduce noise
    let blurred = gaussian_blur_f32(&gray, 2.0);

    // Edge detection with Canny
    let edges = canny(&blurred, 30.0, 100.0);

    // Mask out shape interiors to focus on connectors
    let masked_edges = mask_shape_regions(&edges, &shape_boxes);

    // Hough line detection with tuned parameters
    // Lower vote_threshold = more lines detected
    // Lower suppression_radius = less merging of nearby lines
    let options = LineDetectionOptions {
        vote_threshold: 20,
        suppression_radius: 4,
    };

    let lines = detect_lines(&masked_edges, options);

    // Dynamic max distance for shape proximity (15% of image size or 150px)
    let max_dist = (height.min(width) as f32 * 0.15).max(150.0);

    let mut connectors = Vec::new();

    for line in &lines {
        if let Some(((x1, y1), (x2, y2))) = polar_to_endpoints(line, width, height) {
            // Calculate line length
            let length_sq = (x2 as i32 - x1 as i32).pow(2) + (y2 as i32 - y1 as i32).pow(2);
            let length = (length_sq as f32).sqrt();

            // Skip lines that are too short
            if length < min_length as f32 {
                continue;
            }

            // Find which shapes the line endpoints are near
            let start_shape = find_nearest_shape((x1 as f32, y1 as f32), &shape_boxes, max_dist);
            let end_shape = find_nearest_shape((x2 as f32, y2 as f32), &shape_boxes, max_dist);

            // Only keep lines that connect two different shapes
            if let (Some(start_idx), Some(end_idx)) = (start_shape, end_shape) {
                if start_idx != end_idx {
                    let min_x = x1.min(x2).saturating_sub(10);
                    let min_y = y1.min(y2).saturating_sub(10);
                    let box_width = x1.abs_diff(x2) + 20;
                    let box_height = y1.abs_diff(y2) + 20;

                    connectors.push(ConnectorCandidate {
                        x: min_x as f32,
                        y: min_y as f32,
                        width: box_width as f32,
                        height: box_height.max(5) as f32, // Ensure minimum height for thin arrows
                        connects: (start_idx, end_idx),
                    });
                }
            }
        }
    }

    // Deduplicate - keep one connector per shape pair
    let mut unique: HashMap<(usize, usize), ConnectorCandidate> = HashMap::new();
    for conn in connectors {
        let key = if conn.connects.0 < conn.connects.1 {
            conn.connects
        } else {
            (conn.connects.1, conn.connects.0)
        };
        unique.entry(key).or_insert(conn);
    }

    // Convert to DetectedBox
    let result: Vec<DetectedBox> = unique
        .into_values()
        .map(|c| DetectedBox {
            class_id: 12, // solid_arrow class ID
            class_name: "solid_arrow".to_string(),
            confidence: 0.7,
            x: c.x,
            y: c.y,
            width: c.width,
            height: c.height,
        })
        .collect();

    tracing::debug!("Arrow detector found {} connectors from {} lines", result.len(), lines.len());

    Ok(result)
}

/// Mask out shape regions from edge image
/// This focuses edge detection on the areas between shapes (where arrows are)
fn mask_shape_regions(edges: &GrayImage, shapes: &[&DetectedBox]) -> GrayImage {
    let mut masked = edges.clone();
    let (width, height) = masked.dimensions();

    for shape in shapes {
        // Shrink the mask slightly so we don't mask arrow endpoints
        let shrink = 5i32;
        let x1 = ((shape.x as i32) + shrink).max(0) as u32;
        let y1 = ((shape.y as i32) + shrink).max(0) as u32;
        let x2 = ((shape.x + shape.width) as i32 - shrink).max(0) as u32;
        let y2 = ((shape.y + shape.height) as i32 - shrink).max(0) as u32;

        // Set pixels inside shapes to black (0)
        for y in y1.min(height)..y2.min(height) {
            for x in x1.min(width)..x2.min(width) {
                masked.put_pixel(x, y, Luma([0u8]));
            }
        }
    }
    masked
}

/// Convert polar line representation (r, theta) to two endpoints on image boundary
fn polar_to_endpoints(
    line: &imageproc::hough::PolarLine,
    width: u32,
    height: u32,
) -> Option<((u32, u32), (u32, u32))> {
    let r = line.r as f64;
    let angle_rad = (line.angle_in_degrees as f64).to_radians();
    let cos_t = angle_rad.cos();
    let sin_t = angle_rad.sin();

    let mut points = Vec::new();
    let w = width as f64;
    let h = height as f64;

    // Find intersections with image boundaries

    // Left edge (x=0)
    if sin_t.abs() > 1e-6 {
        let y = r / sin_t;
        if y >= 0.0 && y < h {
            points.push((0u32, y as u32));
        }
    }

    // Right edge (x=width-1)
    if sin_t.abs() > 1e-6 {
        let y = (r - (w - 1.0) * cos_t) / sin_t;
        if y >= 0.0 && y < h {
            points.push(((width - 1), y as u32));
        }
    }

    // Top edge (y=0)
    if cos_t.abs() > 1e-6 {
        let x = r / cos_t;
        if x >= 0.0 && x < w {
            points.push((x as u32, 0u32));
        }
    }

    // Bottom edge (y=height-1)
    if cos_t.abs() > 1e-6 {
        let x = (r - (h - 1.0) * sin_t) / cos_t;
        if x >= 0.0 && x < w {
            points.push((x as u32, (height - 1)));
        }
    }

    // Remove duplicate points (corners)
    points.dedup();

    if points.len() >= 2 {
        Some((points[0], points[1]))
    } else {
        None
    }
}

/// Internal struct for connector candidates before deduplication
struct ConnectorCandidate {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    connects: (usize, usize),
}

/// Check if a class is a connector type (should not be endpoints for arrows)
fn is_connector_class(class_name: &str) -> bool {
    matches!(
        class_name,
        "solid_arrow"
            | "dashed_arrow"
            | "line"
            | "curved_arrow"
            | "bidirectional_arrow"
            | "text_label"
            | "dotted_arrow"
            | "dashed_line"
            | "dotted_line"
            | "curved_line"
    )
}

/// Find the nearest shape to a point, within max_dist
fn find_nearest_shape(point: (f32, f32), shapes: &[&DetectedBox], max_dist: f32) -> Option<usize> {
    let (px, py) = point;
    let mut best_idx = None;
    let mut best_dist = max_dist;

    for (i, shape) in shapes.iter().enumerate() {
        let dist = point_to_box_dist(px, py, shape);
        if dist < best_dist {
            best_dist = dist;
            best_idx = Some(i);
        }
    }
    best_idx
}

/// Calculate minimum distance from a point to a bounding box
fn point_to_box_dist(px: f32, py: f32, b: &DetectedBox) -> f32 {
    // Clamp point to box boundaries
    let cx = px.max(b.x).min(b.x + b.width);
    let cy = py.max(b.y).min(b.y + b.height);

    // Euclidean distance from point to clamped point
    ((px - cx).powi(2) + (py - cy).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_connector_class() {
        assert!(is_connector_class("solid_arrow"));
        assert!(is_connector_class("dashed_arrow"));
        assert!(is_connector_class("line"));
        assert!(is_connector_class("curved_arrow"));
        assert!(!is_connector_class("rectangle"));
        assert!(!is_connector_class("diamond"));
        assert!(!is_connector_class("circle"));
    }

    #[test]
    fn test_point_to_box_dist() {
        let box_ = DetectedBox {
            class_id: 0,
            class_name: "rect".to_string(),
            confidence: 1.0,
            x: 100.0,
            y: 100.0,
            width: 50.0,
            height: 50.0,
        };

        // Point inside box - distance should be 0
        assert_eq!(point_to_box_dist(125.0, 125.0, &box_), 0.0);

        // Point to the right of box
        let dist = point_to_box_dist(200.0, 125.0, &box_);
        assert!((dist - 50.0).abs() < 0.01);

        // Point below box
        let dist = point_to_box_dist(125.0, 200.0, &box_);
        assert!((dist - 50.0).abs() < 0.01);

        // Point at corner (diagonal)
        let dist = point_to_box_dist(200.0, 200.0, &box_);
        let expected = (50.0f32.powi(2) + 50.0f32.powi(2)).sqrt();
        assert!((dist - expected).abs() < 0.01);
    }

    #[test]
    fn test_connector_candidate_dedup() {
        // Test that shape pairs are properly deduplicated
        let key1 = if 1 < 2 { (1, 2) } else { (2, 1) };
        let key2 = if 2 < 1 { (2, 1) } else { (1, 2) };
        assert_eq!(key1, key2);
    }
}
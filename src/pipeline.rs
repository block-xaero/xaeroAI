//! Whiteboard to Mermaid pipeline
//!
//! Orchestrates: YOLO (shapes) → TrOCR (text) → Dictionary (correct) → Phi+LoRA (mermaid)

use crate::dictionary::{Dictionary, DictionaryBuilder, DomainSource};
use crate::runtime::{DetectedBox, InferenceInput, InferenceOutput, Runtime};
use crate::skill::Skill;
use anyhow::{anyhow, Result};
use base64::Engine;
use image::GenericImageView;
use std::path::Path;

/// A detected shape with its text content
#[derive(Debug, Clone)]
pub struct DetectedShape {
    pub id: usize,
    pub shape_type: String,
    pub confidence: f32,
    pub bbox: BoundingBox,
    pub text: Option<String>,
    pub connects_to: Vec<usize>,
}

/// Bounding box
#[derive(Debug, Clone)]
pub struct BoundingBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl BoundingBox {
    pub fn center(&self) -> (f32, f32) {
        (self.x + self.width / 2.0, self.y + self.height / 2.0)
    }
}

/// Pipeline result
#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub shapes: Vec<DetectedShape>,
    pub mermaid: String,
    pub diagram_type: DiagramType,
    pub timing: PipelineTiming,
}

/// Diagram type inferred from shapes
#[derive(Debug, Clone, PartialEq)]
pub enum DiagramType {
    Flowchart,
    Sequence,
    Class,
    State,
    ER,
    Unknown,
}

/// Processing timing breakdown
#[derive(Debug, Clone, Default)]
pub struct PipelineTiming {
    pub detection_ms: u64,
    pub ocr_ms: u64,
    pub layout_ms: u64,
    pub generation_ms: u64,
    pub total_ms: u64,
}

/// Whiteboard processing pipeline
pub struct WhiteboardPipeline {
    runtime: Runtime,
    yolo_model: String,
    ocr_model: String,
    phi_model: String,
    dictionary: Dictionary,
    ocr_dict: Vec<String>,  // Character vocabulary for CTC decode
}

impl WhiteboardPipeline {
    /// Create a new pipeline
    pub fn new(yolo_dir: &Path, ocr_dir: &Path, phi_dir: &Path) -> Result<Self> {
        let mut runtime = Runtime::new()?;

        let yolo_skill = Skill::load(yolo_dir)?;
        let yolo_model = yolo_skill.name.clone();
        runtime.load_from_skill(&yolo_skill, yolo_dir)?;

        let ocr_skill = Skill::load(ocr_dir)?;
        let ocr_model = ocr_skill.name.clone();
        runtime.load_from_skill(&ocr_skill, ocr_dir)?;
        
        // Load OCR dictionary for CTC decoding
        let dict_path = ocr_dir.join("dict.txt");
        let ocr_dict = if dict_path.exists() {
            let content = std::fs::read_to_string(&dict_path)?;
            // First char is blank for CTC
            let mut chars: Vec<String> = vec!["".to_string()];
            chars.extend(content.lines().map(|s| s.to_string()));
            chars
        } else {
            // Fallback: basic ASCII
            let mut chars: Vec<String> = vec!["".to_string()];
            chars.extend((32u8..127).map(|c| (c as char).to_string()));
            chars
        };

        let phi_skill = Skill::load(phi_dir)?;
        let phi_model = phi_skill.name.clone();
        runtime.load_from_skill(&phi_skill, phi_dir)?;

        let dictionary = DictionaryBuilder::new().with_common_terms().build();

        Ok(Self {
            runtime,
            yolo_model,
            ocr_model,
            phi_model,
            dictionary,
            ocr_dict,
        })
    }

    /// Add terms to dictionary from current diagram context
    pub fn add_context_terms(&mut self, terms: &[&str]) {
        self.dictionary.add_terms(terms, DomainSource::DiagramLabels);
    }

    /// Add user correction to dictionary
    pub fn add_correction(&mut self, wrong: &str, right: &str) {
        self.dictionary.add_correction(wrong, right);
    }

    /// Process whiteboard image to mermaid
    pub fn process(&mut self, image_data: &[u8]) -> Result<PipelineResult> {
        let start = std::time::Instant::now();
        let mut timing = PipelineTiming::default();

        // Load image once for cropping later
        let img = image::load_from_memory(image_data)?;

        // Step 1: Detect shapes with YOLO
        let detect_start = std::time::Instant::now();
        let boxes = self.detect_shapes(image_data)?;
        timing.detection_ms = detect_start.elapsed().as_millis() as u64;

        // Step 2: Run OCR on text regions
        let ocr_start = std::time::Instant::now();
        let mut shapes = self.extract_text(&img, &boxes)?;
        timing.ocr_ms = ocr_start.elapsed().as_millis() as u64;

        // Step 3: Analyze layout (pure Rust, no ML)
        let layout_start = std::time::Instant::now();
        self.analyze_connections(&mut shapes, &boxes);
        let diagram_type = self.infer_diagram_type(&shapes);
        timing.layout_ms = layout_start.elapsed().as_millis() as u64;

        // Step 4: Generate Mermaid with Phi
        let gen_start = std::time::Instant::now();
        let mermaid = self.generate_mermaid(&shapes, &diagram_type)?;
        timing.generation_ms = gen_start.elapsed().as_millis() as u64;

        timing.total_ms = start.elapsed().as_millis() as u64;

        Ok(PipelineResult {
            shapes,
            mermaid,
            diagram_type,
            timing,
        })
    }

    fn detect_shapes(&mut self, image_data: &[u8]) -> Result<Vec<DetectedBox>> {
        let input = InferenceInput::Image {
            data_base64: base64::engine::general_purpose::STANDARD.encode(image_data),
        };
        let output = self.runtime.infer_sync(&self.yolo_model, input)?;

        match output {
            InferenceOutput::Boxes { detections } => Ok(detections),
            _ => Err(anyhow!("YOLO returned unexpected output type")),
        }
    }

    fn extract_text(
        &mut self,
        img: &image::DynamicImage,
        boxes: &[DetectedBox],
    ) -> Result<Vec<DetectedShape>> {
        let mut shapes = Vec::new();
        let (img_w, img_h) = img.dimensions();

        for (id, box_) in boxes.iter().enumerate() {
            let text = if Self::is_text_container(&box_.class_name) {
                // Crop the region for this shape
                match self.crop_and_ocr(img, box_, img_w, img_h) {
                    Ok(Some(text)) => Some(text),
                    Ok(None) => None,
                    Err(e) => {
                        tracing::warn!("OCR failed for shape {}: {}", id, e);
                        None
                    }
                }
            } else {
                None
            };

            shapes.push(DetectedShape {
                id,
                shape_type: box_.class_name.clone(),
                confidence: box_.confidence,
                bbox: BoundingBox {
                    x: box_.x,
                    y: box_.y,
                    width: box_.width,
                    height: box_.height,
                },
                text,
                connects_to: Vec::new(),
            });
        }

        Ok(shapes)
    }

    /// Crop a shape region and run OCR on it
    fn crop_and_ocr(
        &mut self,
        img: &image::DynamicImage,
        box_: &DetectedBox,
        img_w: u32,
        img_h: u32,
    ) -> Result<Option<String>> {
        // Clamp coordinates to image bounds
        let x = (box_.x.max(0.0) as u32).min(img_w.saturating_sub(1));
        let y = (box_.y.max(0.0) as u32).min(img_h.saturating_sub(1));
        let w = (box_.width as u32).min(img_w.saturating_sub(x)).max(1);
        let h = (box_.height as u32).min(img_h.saturating_sub(y)).max(1);

        // Skip tiny regions
        if w < 10 || h < 10 {
            return Ok(None);
        }

        // Crop
        let cropped = img.crop_imm(x, y, w, h);

        // Encode as PNG
        let mut png_bytes = Vec::new();
        cropped.write_to(
            &mut std::io::Cursor::new(&mut png_bytes),
            image::ImageFormat::Png,
        )?;

        // Run PaddleOCR recognition
        let input = InferenceInput::Image {
            data_base64: base64::engine::general_purpose::STANDARD.encode(&png_bytes),
        };
        let output = self.runtime.infer_sync(&self.ocr_model, input)?;

        match output {
            InferenceOutput::Text { content } => {
                let trimmed = content.trim();
                if trimmed.is_empty() {
                    Ok(None)
                } else {
                    // Apply dictionary correction
                    let corrected = self.dictionary.correct_phrase(trimmed);
                    Ok(Some(corrected.corrected))
                }
            }
            // PaddleOCR returns CTC logits that need decoding
            InferenceOutput::Json { data } => {
                // Decode CTC output if runtime returns raw logits
                if let Some(logits) = data.get("logits") {
                    let text = self.ctc_greedy_decode(logits)?;
                    if text.is_empty() {
                        Ok(None)
                    } else {
                        let corrected = self.dictionary.correct_phrase(&text);
                        Ok(Some(corrected.corrected))
                    }
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    /// CTC greedy decode: take argmax at each timestep, collapse repeats, remove blanks
    fn ctc_greedy_decode(&self, logits: &serde_json::Value) -> Result<String> {
        let arr = logits.as_array()
            .ok_or_else(|| anyhow!("CTC logits not an array"))?;
        
        let mut result = String::new();
        let mut prev_idx: Option<usize> = None;
        
        for timestep in arr {
            let probs = timestep.as_array()
                .ok_or_else(|| anyhow!("Timestep not an array"))?;
            
            // Find argmax
            let (max_idx, _) = probs.iter().enumerate()
                .map(|(i, v)| (i, v.as_f64().unwrap_or(f64::NEG_INFINITY)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, 0.0));
            
            // Skip if same as previous (collapse repeats)
            if Some(max_idx) == prev_idx {
                continue;
            }
            prev_idx = Some(max_idx);
            
            // Skip blank (index 0)
            if max_idx == 0 {
                continue;
            }
            
            // Map to character
            if let Some(ch) = self.ocr_dict.get(max_idx) {
                result.push_str(ch);
            }
        }
        
        Ok(result)
    }

    fn is_text_container(class: &str) -> bool {
        matches!(
            class,
            "rectangle"
                | "rounded_rectangle"
                | "oval"
                | "circle"
                | "diamond"
                | "hexagon"
                | "parallelogram"
                | "sticky_note"
                | "text_label"
                | "cloud"
                | "cylinder"
                | "square"
                | "ellipse"
                | "document_shape"
                | "arrow_box"
        )
    }

    fn analyze_connections(&self, shapes: &mut [DetectedShape], boxes: &[DetectedBox]) {
        // Find arrow indices
        let arrow_indices: Vec<usize> = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| Self::is_arrow(&b.class_name))
            .map(|(i, _)| i)
            .collect();

        let non_arrow_indices: Vec<usize> = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| !Self::is_arrow(&b.class_name))
            .map(|(i, _)| i)
            .collect();

        // For each arrow, find what it connects
        for arrow_idx in arrow_indices {
            let arrow = &boxes[arrow_idx];
            
            // Arrow endpoints: start = left side, end = right side
            let (start_x, start_y) = (arrow.x, arrow.y + arrow.height / 2.0);
            let (end_x, end_y) = (arrow.x + arrow.width, arrow.y + arrow.height / 2.0);

            let mut closest_to_start: Option<(usize, f32)> = None;
            let mut closest_to_end: Option<(usize, f32)> = None;

            for &shape_idx in &non_arrow_indices {
                let shape = &boxes[shape_idx];
                let center = (shape.x + shape.width / 2.0, shape.y + shape.height / 2.0);

                let dist_to_start = Self::distance(start_x, start_y, center.0, center.1);
                let dist_to_end = Self::distance(end_x, end_y, center.0, center.1);

                // Only consider shapes within reasonable distance
                let max_dist = 150.0;

                if dist_to_start < max_dist {
                    if closest_to_start.map(|(_, d)| dist_to_start < d).unwrap_or(true) {
                        closest_to_start = Some((shape_idx, dist_to_start));
                    }
                }
                if dist_to_end < max_dist {
                    if closest_to_end.map(|(_, d)| dist_to_end < d).unwrap_or(true) {
                        closest_to_end = Some((shape_idx, dist_to_end));
                    }
                }
            }

            // Record connection: start_shape → end_shape
            if let (Some((start_shape, _)), Some((end_shape, _))) = (closest_to_start, closest_to_end)
            {
                if start_shape != end_shape && start_shape < shapes.len() && end_shape < shapes.len() {
                    shapes[start_shape].connects_to.push(end_shape);
                }
            }
        }
    }

    fn is_arrow(class: &str) -> bool {
        matches!(
            class,
            "solid_arrow"
                | "dashed_arrow"
                | "bidirectional_arrow"
                | "curved_arrow"
                | "dotted_arrow"
                | "dotted_line"
                | "curved_line"
                | "curved_bidirectional_arrow"
        )
    }

    fn distance(x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
        ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt()
    }

    fn infer_diagram_type(&self, shapes: &[DetectedShape]) -> DiagramType {
        let has_diamonds = shapes.iter().any(|s| s.shape_type == "diamond");
        let has_cylinders = shapes.iter().any(|s| s.shape_type == "cylinder" || s.shape_type == "database_icon");
        let has_stick_figures = shapes.iter().any(|s| s.shape_type == "stick_figure");
        let has_ovals = shapes
            .iter()
            .any(|s| s.shape_type == "oval" || s.shape_type == "ellipse");

        if has_stick_figures {
            DiagramType::Sequence
        } else if has_cylinders {
            DiagramType::ER
        } else if has_diamonds {
            DiagramType::Flowchart
        } else if has_ovals && !has_diamonds {
            DiagramType::State
        } else {
            DiagramType::Flowchart
        }
    }

    fn generate_mermaid(&mut self, shapes: &[DetectedShape], diagram_type: &DiagramType) -> Result<String> {
        let prompt = self.build_prompt(shapes, diagram_type);

        let input = InferenceInput::Text { prompt };
        let output = self.runtime.infer_sync(&self.phi_model, input)?;

        match output {
            InferenceOutput::Text { content } => Ok(Self::extract_mermaid_code(&content)),
            _ => Err(anyhow!("Phi returned unexpected output type")),
        }
    }

    fn build_prompt(&self, shapes: &[DetectedShape], diagram_type: &DiagramType) -> String {
        let type_str = match diagram_type {
            DiagramType::Flowchart => "flowchart TD",
            DiagramType::Sequence => "sequenceDiagram",
            DiagramType::Class => "classDiagram",
            DiagramType::State => "stateDiagram-v2",
            DiagramType::ER => "erDiagram",
            DiagramType::Unknown => "flowchart TD",
        };

        let mut prompt = String::from("<|user|>\n");
        prompt.push_str(&format!(
            "Convert this whiteboard to a Mermaid {} diagram.\n\nShapes detected:\n",
            type_str
        ));

        // List non-arrow shapes
        for shape in shapes.iter().filter(|s| !Self::is_arrow(&s.shape_type)) {
            let text = shape.text.as_deref().unwrap_or("[no text]");
            prompt.push_str(&format!(
                "- Shape {}: {} containing \"{}\"\n",
                shape.id, shape.shape_type, text
            ));
        }

        // List connections
        prompt.push_str("\nConnections:\n");
        let mut has_connections = false;
        for shape in shapes {
            for &target_id in &shape.connects_to {
                if let Some(target) = shapes.iter().find(|s| s.id == target_id) {
                    let src_fallback = format!("Shape{}", shape.id);
                    let dst_fallback = format!("Shape{}", target.id);
                    let src_text = shape.text.as_deref().unwrap_or(&src_fallback);
                    let dst_text = target.text.as_deref().unwrap_or(&dst_fallback);
                    prompt.push_str(&format!("- \"{}\" → \"{}\"\n", src_text, dst_text));
                    has_connections = true;
                }
            }
        }
        if !has_connections {
            prompt.push_str("- (no connections detected)\n");
        }

        prompt.push_str(&format!(
            "\nGenerate valid Mermaid {} syntax:\n<|end|>\n<|assistant|>\n```mermaid\n{}\n",
            type_str, type_str
        ));

        prompt
    }

    fn extract_mermaid_code(response: &str) -> String {
        // Try to find ```mermaid block
        if let Some(start) = response.find("```mermaid") {
            let content_start = start + "```mermaid".len();
            if let Some(end) = response[content_start..].find("```") {
                return response[content_start..content_start + end].trim().to_string();
            }
        }

        // Try to find any ``` block
        if let Some(start) = response.find("```") {
            let content_start = start + 3;
            // Skip language identifier line
            let content_start = response[content_start..]
                .find('\n')
                .map(|i| content_start + i + 1)
                .unwrap_or(content_start);

            if let Some(end) = response[content_start..].find("```") {
                return response[content_start..content_start + end].trim().to_string();
            }
        }

        // Return as-is if no code block found
        response.trim().to_string()
    }

    /// Swap LoRA adapter for different diagram styles
    pub fn swap_lora(&mut self, lora_path: &Path) -> Result<()> {
        self.runtime.swap_lora(&self.phi_model, lora_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounding_box_center() {
        let bbox = BoundingBox {
            x: 100.0,
            y: 100.0,
            width: 200.0,
            height: 100.0,
        };
        assert_eq!(bbox.center(), (200.0, 150.0));
    }

    #[test]
    fn test_extract_mermaid_code() {
        let response = "Here's the diagram:\n```mermaid\nflowchart TD\n  A --> B\n```\nDone!";
        let code = WhiteboardPipeline::extract_mermaid_code(response);
        assert_eq!(code, "flowchart TD\n  A --> B");
    }

    #[test]
    fn test_extract_mermaid_code_no_fence() {
        let response = "flowchart TD\n  A --> B";
        let code = WhiteboardPipeline::extract_mermaid_code(response);
        assert_eq!(code, "flowchart TD\n  A --> B");
    }

    #[test]
    fn test_is_arrow() {
        assert!(WhiteboardPipeline::is_arrow("solid_arrow"));
        assert!(WhiteboardPipeline::is_arrow("dashed_arrow"));
        assert!(WhiteboardPipeline::is_arrow("dotted_line"));
        assert!(!WhiteboardPipeline::is_arrow("rectangle"));
    }

    #[test]
    fn test_is_text_container() {
        assert!(WhiteboardPipeline::is_text_container("rectangle"));
        assert!(WhiteboardPipeline::is_text_container("diamond"));
        assert!(WhiteboardPipeline::is_text_container("document_shape"));
        assert!(!WhiteboardPipeline::is_text_container("solid_arrow"));
    }

    #[test]
    fn test_distance() {
        assert_eq!(WhiteboardPipeline::distance(0.0, 0.0, 3.0, 4.0), 5.0);
    }
}

//! Model runtime - loading and inference for GGUF and ONNX models
//!
//! GGUF (Phi, Llama) via llama-cpp-2
//! ONNX (YOLO, PaddleOCR) via ort

use crate::skill::{ModelKind, Skill};
use anyhow::{anyhow, Result};
use image::GenericImageView;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::OnceLock;

// Global llama backend (must be initialized once)
static LLAMA_BACKEND: OnceLock<LlamaBackend> = OnceLock::new();

fn get_llama_backend() -> &'static LlamaBackend {
    LLAMA_BACKEND.get_or_init(|| LlamaBackend::init().expect("Failed to init llama backend"))
}

/// Inference input (serializable for FFI)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum InferenceInput {
    /// Text prompt
    Text { prompt: String },
    /// Image data (base64 encoded for JSON transport)
    Image { data_base64: String },
    /// Structured JSON input
    Json { data: serde_json::Value },
}

/// Inference output (serializable for FFI)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum InferenceOutput {
    /// Generated text
    Text { content: String },
    /// Detected boxes (for YOLO)
    Boxes { detections: Vec<DetectedBox> },
    /// Structured JSON output
    Json { data: serde_json::Value },
}

/// Bounding box detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedBox {
    pub class_id: u32,
    pub class_name: String,
    pub confidence: f32,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// Loaded model instance
enum LoadedModel {
    Gguf(GgufModel),
    Onnx(OnnxModel),
}

/// GGUF model wrapper
struct GgufModel {
    name: String,
    model: LlamaModel,
    #[allow(dead_code)]
    active_lora: Option<String>,
}

/// ONNX model type
#[derive(Debug, Clone, PartialEq)]
enum OnnxModelType {
    Detection,   // YOLO - outputs boxes
    Recognition, // OCR - outputs text/logits
}

/// ONNX model wrapper
struct OnnxModel {
    name: String,
    session: ort::session::Session,
    model_type: OnnxModelType,
    class_names: Option<Vec<String>>,
    char_dict: Option<Vec<String>>,  // For OCR CTC decode
    input_height: u32,
    input_width: u32,
}

/// Runtime manages loaded models and inference
pub struct Runtime {
    models: HashMap<String, LoadedModel>,
}

impl Runtime {
    pub fn new() -> Result<Self> {
        // Initialize llama backend
        let _ = get_llama_backend();

        Ok(Self {
            models: HashMap::new(),
        })
    }

    pub fn load_from_skill(&mut self, skill: &Skill, models_dir: &Path) -> Result<()> {
        let model_file = skill
            .model_file
            .as_ref()
            .ok_or_else(|| anyhow!("No model file specified in skill"))?;

        let model_path = models_dir.join(model_file);

        if !model_path.exists() {
            return Err(anyhow!("Model file not found: {:?}", model_path));
        }

        let loaded = match skill.kind {
            ModelKind::Gguf => self.load_gguf(&skill.name, &model_path)?,
            ModelKind::Onnx => self.load_onnx(&skill.name, &model_path, models_dir)?,
            _ => return Err(anyhow!("Unsupported model kind: {:?}", skill.kind)),
        };

        self.models.insert(skill.name.clone(), loaded);
        tracing::info!("Loaded model: {}", skill.name);

        Ok(())
    }

    fn load_gguf(&mut self, name: &str, model_path: &Path) -> Result<LoadedModel> {
        tracing::info!("Loading GGUF model: {:?}", model_path);

        let params = LlamaModelParams::default();
        let model = LlamaModel::load_from_file(get_llama_backend(), model_path, &params)
            .map_err(|e| anyhow!("Failed to load GGUF model: {:?}", e))?;

        Ok(LoadedModel::Gguf(GgufModel {
            name: name.to_string(),
            model,
            active_lora: None,
        }))
    }

    fn load_onnx(&mut self, name: &str, model_path: &Path, models_dir: &Path) -> Result<LoadedModel> {
        tracing::info!("Loading ONNX model: {:?}", model_path);

        let session = ort::session::Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .commit_from_file(model_path)?;

        // Detect model type based on files present
        let classes_path = models_dir.join("classes.txt");
        let dict_path = models_dir.join("dict.txt");
        
        let (model_type, class_names, char_dict, input_height, input_width) = 
            if dict_path.exists() {
                // OCR recognition model (PaddleOCR)
                let content = std::fs::read_to_string(&dict_path)?;
                let mut chars: Vec<String> = vec!["".to_string()]; // blank for CTC
                chars.extend(content.lines().map(|s| s.to_string()));
                tracing::info!("Loaded OCR dictionary with {} chars", chars.len());
                
                // PaddleOCR rec uses height=32, variable width
                (OnnxModelType::Recognition, None, Some(chars), 48, 320)
            } else if classes_path.exists() {
                // Detection model (YOLO)
                let content = std::fs::read_to_string(&classes_path)?;
                let classes: Vec<String> = content.lines().map(|s| s.trim().to_string()).collect();
                tracing::info!("Loaded {} class names", classes.len());
                
                (OnnxModelType::Detection, Some(classes), None, 640, 640)
            } else {
                // Default to detection
                (OnnxModelType::Detection, None, None, 640, 640)
            };

        Ok(LoadedModel::Onnx(OnnxModel {
            name: name.to_string(),
            session,
            model_type,
            class_names,
            char_dict,
            input_height,
            input_width,
        }))
    }

    pub fn unload(&mut self, model_id: &str) -> Result<()> {
        self.models
            .remove(model_id)
            .ok_or_else(|| anyhow!("Model not loaded: {}", model_id))?;
        tracing::info!("Unloaded model: {}", model_id);
        Ok(())
    }

    pub fn swap_lora(&mut self, base_model_id: &str, lora_path: &Path) -> Result<()> {
        let model = self
            .models
            .get_mut(base_model_id)
            .ok_or_else(|| anyhow!("Base model not loaded: {}", base_model_id))?;

        match model {
            LoadedModel::Gguf(gguf) => {
                tracing::info!("Swapping LoRA adapter: {:?}", lora_path);
                
                // Initialize LoRA adapter
                let _adapter = gguf.model
                    .lora_adapter_init(lora_path)
                    .map_err(|e| anyhow!("Failed to load LoRA: {:?}", e))?;

                gguf.active_lora = Some(lora_path.to_string_lossy().to_string());
                Ok(())
            }
            LoadedModel::Onnx(_) => Err(anyhow!("Cannot apply LoRA to ONNX model")),
        }
    }

    pub fn infer_sync(&mut self, model_id: &str, input: InferenceInput) -> Result<InferenceOutput> {
        let model = self
            .models
            .get_mut(model_id)
            .ok_or_else(|| anyhow!("Model not loaded: {}", model_id))?;

        match model {
            LoadedModel::Gguf(gguf) => Self::infer_gguf_static(gguf, input),
            LoadedModel::Onnx(onnx) => Self::infer_onnx_static(onnx, input),
        }
    }

    fn infer_gguf_static(model: &GgufModel, input: InferenceInput) -> Result<InferenceOutput> {
        let prompt = match input {
            InferenceInput::Text { prompt } => prompt,
            InferenceInput::Json { data } => serde_json::to_string(&data)?,
            InferenceInput::Image { .. } => {
                return Err(anyhow!("GGUF models don't support image input"));
            }
        };

        // Create context
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(2048));
        
        let mut ctx = model
            .model
            .new_context(get_llama_backend(), ctx_params)
            .map_err(|e| anyhow!("Failed to create context: {:?}", e))?;

        // Tokenize input
        let tokens = model
            .model
            .str_to_token(&prompt, llama_cpp_2::model::AddBos::Always)
            .map_err(|e| anyhow!("Tokenization failed: {:?}", e))?;

        // Create batch and add tokens
        let mut batch = LlamaBatch::new(2048, 1);
        
        for (i, token) in tokens.iter().enumerate() {
            let is_last = i == tokens.len() - 1;
            batch.add(*token, i as i32, &[0], is_last)
                .map_err(|e| anyhow!("Failed to add token to batch: {:?}", e))?;
        }

        // Decode the batch
        ctx.decode(&mut batch)
            .map_err(|e| anyhow!("Decode failed: {:?}", e))?;

        // Create sampler chain for temperature sampling
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(0.7),
            LlamaSampler::dist(42), // seed
        ]);

        // Generate tokens
        let mut output_tokens = Vec::new();
        let max_tokens = 512;
        let mut n_cur = tokens.len();

        for _ in 0..max_tokens {
            // Sample next token
            let new_token = sampler.sample(&ctx, batch.n_tokens() - 1);

            // Check for EOS
            if model.model.is_eog_token(new_token) {
                break;
            }

            output_tokens.push(new_token);

            // Prepare next batch
            batch.clear();
            batch.add(new_token, n_cur as i32, &[0], true)
                .map_err(|e| anyhow!("Failed to add token: {:?}", e))?;
            n_cur += 1;

            ctx.decode(&mut batch)
                .map_err(|e| anyhow!("Decode failed: {:?}", e))?;
        }

        // Convert tokens to string
        let output = output_tokens
            .iter()
            .filter_map(|t| model.model.token_to_str(*t, llama_cpp_2::model::Special::Tokenize).ok())
            .collect::<String>();

        Ok(InferenceOutput::Text { content: output })
    }

    fn infer_onnx_static(model: &mut OnnxModel, input: InferenceInput) -> Result<InferenceOutput> {
        match (&model.model_type, input) {
            // Detection model (YOLO)
            (OnnxModelType::Detection, InferenceInput::Image { data_base64 }) => {
                Self::infer_onnx_detection_static(model, &data_base64)
            }
            // Recognition model (PaddleOCR)
            (OnnxModelType::Recognition, InferenceInput::Image { data_base64 }) => {
                Self::infer_onnx_recognition_static(model, &data_base64)
            }
            _ => Err(anyhow!("Unsupported input type for ONNX model")),
        }
    }

    /// YOLO detection inference
    fn infer_onnx_detection_static(model: &mut OnnxModel, data_base64: &str) -> Result<InferenceOutput> {
        use base64::Engine;
        
        // Decode base64 to bytes
        let image_bytes = base64::engine::general_purpose::STANDARD
            .decode(data_base64)?;

        // Decode image
        let img = image::load_from_memory(&image_bytes)?;
        let (orig_width, orig_height) = img.dimensions();

        // Resize to model input size (640x640 for YOLO)
        let resized = img.resize_exact(
            model.input_width,
            model.input_height,
            image::imageops::FilterType::Triangle,
        );

        // Convert to RGB and normalize to [0, 1]
        let rgb = resized.to_rgb8();

        // Create input tensor [1, 3, H, W] - CHW format, normalized
        let mut input_data = vec![0.0f32; 3 * model.input_height as usize * model.input_width as usize];
        
        for y in 0..model.input_height as usize {
            for x in 0..model.input_width as usize {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                let idx = y * model.input_width as usize + x;
                input_data[0 * model.input_height as usize * model.input_width as usize + idx] = pixel[0] as f32 / 255.0;
                input_data[1 * model.input_height as usize * model.input_width as usize + idx] = pixel[1] as f32 / 255.0;
                input_data[2 * model.input_height as usize * model.input_width as usize + idx] = pixel[2] as f32 / 255.0;
            }
        }

        // Create ONNX tensor
        let input_tensor = ort::value::Tensor::from_array((
            [1usize, 3, model.input_height as usize, model.input_width as usize],
            input_data.into_boxed_slice(),
        ))?;

        // Run inference
        let outputs = model.session.run(ort::inputs![input_tensor])?;

        // Parse YOLO output - use try_extract_tensor on DynValue
        let (shape, raw_data) = outputs[0].try_extract_tensor::<f32>()?;

        let mut detections = Vec::new();
        let num_classes = model.class_names.as_ref().map(|c| c.len()).unwrap_or(80);
        // Shape is [1, 4+num_classes, num_boxes] - convert i64 to usize
        let num_boxes = shape.get(2).map(|&x| x as usize).unwrap_or(8400);
        
        for i in 0..num_boxes {
            let x_center = raw_data[0 * num_boxes + i];
            let y_center = raw_data[1 * num_boxes + i];
            let w = raw_data[2 * num_boxes + i];
            let h = raw_data[3 * num_boxes + i];

            let mut best_class = 0usize;
            let mut best_conf = 0.0f32;
            for c in 0..num_classes {
                let conf = raw_data[(4 + c) * num_boxes + i];
                if conf > best_conf {
                    best_conf = conf;
                    best_class = c;
                }
            }

            if best_conf > 0.25 {
                let scale_x = orig_width as f32 / model.input_width as f32;
                let scale_y = orig_height as f32 / model.input_height as f32;

                let x = (x_center - w / 2.0) * scale_x;
                let y = (y_center - h / 2.0) * scale_y;
                let width = w * scale_x;
                let height = h * scale_y;

                let class_name = model
                    .class_names
                    .as_ref()
                    .and_then(|c| c.get(best_class).cloned())
                    .unwrap_or_else(|| format!("class_{}", best_class));

                detections.push(DetectedBox {
                    class_id: best_class as u32,
                    class_name,
                    confidence: best_conf,
                    x,
                    y,
                    width,
                    height,
                });
            }
        }

        // Apply NMS
        detections = Self::nms(detections, 0.45);

        Ok(InferenceOutput::Boxes { detections })
    }

    /// PaddleOCR recognition inference with CTC decode
    fn infer_onnx_recognition_static(model: &mut OnnxModel, data_base64: &str) -> Result<InferenceOutput> {
        use base64::Engine;
        
        // Decode base64 to bytes
        let image_bytes = base64::engine::general_purpose::STANDARD
            .decode(data_base64)?;

        // Decode image
        let img = image::load_from_memory(&image_bytes)?;
        let (orig_width, orig_height) = img.dimensions();

        // PaddleOCR rec expects: height=32, variable width (aspect ratio preserved)
        let target_height = model.input_height;
        let aspect = orig_width as f32 / orig_height as f32;
        let target_width = ((target_height as f32 * aspect).round() as u32).max(32).min(320);

        let resized = img.resize_exact(
            target_width,
            target_height,
            image::imageops::FilterType::Triangle,
        );

        let rgb = resized.to_rgb8();

        // Create input tensor [1, 3, 32, W] with ImageNet normalization
        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];
        
        let mut input_data = vec![0.0f32; 3 * target_height as usize * target_width as usize];

        for y in 0..target_height as usize {
            for x in 0..target_width as usize {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                let idx = y * target_width as usize + x;
                input_data[0 * target_height as usize * target_width as usize + idx] = (pixel[0] as f32 / 255.0 - mean[0]) / std[0];
                input_data[1 * target_height as usize * target_width as usize + idx] = (pixel[1] as f32 / 255.0 - mean[1]) / std[1];
                input_data[2 * target_height as usize * target_width as usize + idx] = (pixel[2] as f32 / 255.0 - mean[2]) / std[2];
            }
        }

        // Create ONNX tensor
        let input_tensor = ort::value::Tensor::from_array((
            [1usize, 3, target_height as usize, target_width as usize],
            input_data.into_boxed_slice(),
        ))?;

        // Run inference
        let outputs = model.session.run(ort::inputs![input_tensor])?;

        // Output shape is [1, T, vocab_size] where T is sequence length
        let (shape, raw_data) = outputs[0].try_extract_tensor::<f32>()?;

        if shape.len() < 2 {
            return Err(anyhow!("Unexpected OCR output shape: {:?}", shape));
        }

        // Convert i64 shape to usize
        let seq_len = shape[1] as usize;
        let vocab_size = if shape.len() > 2 { shape[2] as usize } else { 0 };

        // CTC greedy decode
        let text = if vocab_size > 0 && model.char_dict.is_some() {
            let dict = model.char_dict.as_ref().unwrap();
            let mut result = String::new();
            let mut prev_idx: Option<usize> = None;

            for t in 0..seq_len {
                // Find argmax - data is [1, T, vocab_size] flattened
                let mut max_idx = 0usize;
                let mut max_val = f32::NEG_INFINITY;
                for v in 0..vocab_size {
                    let idx = t * vocab_size + v;
                    let val = raw_data[idx];
                    if val > max_val {
                        max_val = val;
                        max_idx = v;
                    }
                }

                // Collapse repeats
                if Some(max_idx) == prev_idx {
                    continue;
                }
                prev_idx = Some(max_idx);

                // Skip blank (index 0)
                if max_idx == 0 {
                    continue;
                }

                // Map to character
                if let Some(ch) = dict.get(max_idx) {
                    result.push_str(ch);
                }
            }
            result
        } else {
            // Fallback: return shape info
            format!("[OCR output shape: {:?}]", shape)
        };

        Ok(InferenceOutput::Text { content: text })
    }

    /// Non-maximum suppression
    fn nms(mut boxes: Vec<DetectedBox>, iou_threshold: f32) -> Vec<DetectedBox> {
        // Sort by confidence descending
        boxes.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        let mut keep = Vec::new();

        while !boxes.is_empty() {
            let best = boxes.remove(0);
            keep.push(best.clone());

            boxes.retain(|b| {
                if b.class_id != best.class_id {
                    return true; // Keep different classes
                }
                let iou = Self::compute_iou(&best, b);
                iou < iou_threshold
            });
        }

        keep
    }

    fn compute_iou(a: &DetectedBox, b: &DetectedBox) -> f32 {
        let x1 = a.x.max(b.x);
        let y1 = a.y.max(b.y);
        let x2 = (a.x + a.width).min(b.x + b.width);
        let y2 = (a.y + a.height).min(b.y + b.height);

        let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
        let area_a = a.width * a.height;
        let area_b = b.width * b.height;
        let union = area_a + area_b - intersection;

        if union > 0.0 {
            intersection / union
        } else {
            0.0
        }
    }

    /// Check if a model is loaded
    pub fn is_loaded(&self, model_id: &str) -> bool {
        self.models.contains_key(model_id)
    }

    /// Get list of loaded model IDs
    pub fn loaded_models(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }
}

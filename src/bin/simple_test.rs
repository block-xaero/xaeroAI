//! Simple model test - just load models and verify they work
//!
//! Usage:
//!     cargo run --bin simple_test -- --models-dir ./models
//!
//! Tests each model individually without the full pipeline

use anyhow::{anyhow, Result};
use clap::Parser;
use std::path::PathBuf;
use std::fs;

use xaeroai::{Runtime, Skill, InferenceInput, InferenceOutput};

#[derive(Parser, Debug)]
#[command(name = "simple_test")]
#[command(about = "Test individual models")]
struct Args {
    /// Directory containing model subdirectories
    #[arg(short, long, default_value = "./models")]
    models_dir: PathBuf,

    /// Test YOLO only
    #[arg(long)]
    yolo_only: bool,

    /// Test OCR only
    #[arg(long)]
    ocr_only: bool,

    /// Test Phi only
    #[arg(long)]
    phi_only: bool,

    /// Image for testing YOLO/OCR
    #[arg(long)]
    image: Option<PathBuf>,

    /// Prompt for testing Phi
    #[arg(long)]
    prompt: Option<String>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           xaeroai Simple Model Test                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let mut runtime = Runtime::new()?;

    let test_all = !args.yolo_only && !args.ocr_only && !args.phi_only;

    // Test YOLO
    if test_all || args.yolo_only {
        println!("{}", "─".repeat(60));
        println!("🔍 Testing YOLO (whiteboard-detector)");
        println!("{}", "─".repeat(60));

        let yolo_dir = args.models_dir.join("whiteboard-detector");
        if yolo_dir.exists() {
            let skill = Skill::load(&yolo_dir)?;
            println!("   Loaded skill: {}", skill.name);
            
            runtime.load_from_skill(&skill, &yolo_dir)?;
            println!("   ✅ Model loaded");

            // Test inference if image provided
            if let Some(ref img_path) = args.image {
                let image_data = fs::read(img_path)?;
                let input = InferenceInput::Image {
                    data_base64: base64::Engine::encode(
                        &base64::engine::general_purpose::STANDARD,
                        &image_data
                    ),
                };

                let start = std::time::Instant::now();
                let output = runtime.infer_sync(&skill.name, input)?;
                let elapsed = start.elapsed();

                match output {
                    InferenceOutput::Boxes { detections } => {
                        println!("   ✅ Inference OK ({:?})", elapsed);
                        println!("   Found {} objects:", detections.len());
                        for det in detections.iter().take(5) {
                            println!("      - {} ({:.2})", det.class_name, det.confidence);
                        }
                        if detections.len() > 5 {
                            println!("      ... and {} more", detections.len() - 5);
                        }
                    }
                    _ => println!("   ⚠️ Unexpected output type"),
                }
            } else {
                println!("   (provide --image to test inference)");
            }
        } else {
            println!("   ❌ Not found: {:?}", yolo_dir);
        }
        println!();
    }

    // Test PaddleOCR
    if test_all || args.ocr_only {
        println!("{}", "─".repeat(60));
        println!("📝 Testing PaddleOCR");
        println!("{}", "─".repeat(60));

        let ocr_dir = args.models_dir.join("paddleocr");
        if ocr_dir.exists() {
            let skill = Skill::load(&ocr_dir)?;
            println!("   Loaded skill: {}", skill.name);

            runtime.load_from_skill(&skill, &ocr_dir)?;
            println!("   ✅ Model loaded");

            if let Some(ref img_path) = args.image {
                let image_data = fs::read(img_path)?;
                let input = InferenceInput::Image {
                    data_base64: base64::Engine::encode(
                        &base64::engine::general_purpose::STANDARD,
                        &image_data
                    ),
                };

                let start = std::time::Instant::now();
                let output = runtime.infer_sync(&skill.name, input)?;
                let elapsed = start.elapsed();

                match output {
                    InferenceOutput::Text { content } => {
                        println!("   ✅ Inference OK ({:?})", elapsed);
                        println!("   OCR result: \"{}\"", content);
                    }
                    _ => println!("   ⚠️ Unexpected output type"),
                }
            } else {
                println!("   (provide --image to test inference)");
            }
        } else {
            println!("   ❌ Not found: {:?}", ocr_dir);
        }
        println!();
    }

    // Test Phi
    if test_all || args.phi_only {
        println!("{}", "─".repeat(60));
        println!("🧠 Testing Phi (cyan-lens)");
        println!("{}", "─".repeat(60));

        let phi_dir = args.models_dir.join("cyan-lens");
        if phi_dir.exists() {
            let skill = Skill::load(&phi_dir)?;
            println!("   Loaded skill: {}", skill.name);

            runtime.load_from_skill(&skill, &phi_dir)?;
            println!("   ✅ Model loaded");

            let prompt = args.prompt.clone().unwrap_or_else(|| {
                "<|user|>\nCreate a simple Mermaid flowchart with 3 nodes.\n<|end|>\n<|assistant|>\n```mermaid\nflowchart TD\n".to_string()
            });

            println!("   Prompt: {}...", &prompt.chars().take(50).collect::<String>());

            let input = InferenceInput::Text { prompt };

            let start = std::time::Instant::now();
            let output = runtime.infer_sync(&skill.name, input)?;
            let elapsed = start.elapsed();

            match output {
                InferenceOutput::Text { content } => {
                    println!("   ✅ Inference OK ({:?})", elapsed);
                    println!("   Output ({} chars):", content.len());
                    for line in content.lines().take(10) {
                        println!("      {}", line);
                    }
                }
                _ => println!("   ⚠️ Unexpected output type"),
            }
        } else {
            println!("   ❌ Not found: {:?}", phi_dir);
        }
        println!();
    }

    println!("✅ Done!");
    Ok(())
}

//! Test CLI for the whiteboard-to-mermaid pipeline.
//!
//! Usage:
//!     cargo run --bin test_pipeline -- --image whiteboard.jpg --models-dir ./models
//!
//! This will:
//!     1. Load all three models (YOLO, TrOCR, Phi)
//!     2. Process the image through the pipeline
//!     3. Print each stage's output
//!     4. Save the final mermaid to a file

use anyhow::{anyhow, Result};
use clap::Parser;
use std::path::PathBuf;
use std::fs;
use std::io::Write;

// We'll use the pipeline module
use xaeroai::pipeline::{WhiteboardPipeline, DiagramType};

#[derive(Parser, Debug)]
#[command(name = "test_pipeline")]
#[command(about = "Test the whiteboard-to-mermaid pipeline")]
struct Args {
    /// Path to whiteboard image (PNG or JPEG)
    #[arg(short, long)]
    image: PathBuf,

    /// Directory containing model subdirectories
    #[arg(short, long, default_value = "./models")]
    models_dir: PathBuf,

    /// Output file for mermaid code (optional)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Show verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Skip LLM generation (test detection + OCR only)
    #[arg(long)]
    skip_llm: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Setup logging
    if args.verbose {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .init();
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           xaeroai Pipeline Test CLI                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Check image exists
    if !args.image.exists() {
        return Err(anyhow!("Image not found: {:?}", args.image));
    }
    println!("📸 Image: {:?}", args.image);
    println!("📁 Models: {:?}", args.models_dir);
    println!();

    // Check model directories
    let yolo_dir = args.models_dir.join("whiteboard-detector");
    let ocr_dir = args.models_dir.join("paddleocr");
    let phi_dir = args.models_dir.join("cyan-lens");

    println!("🔍 Checking models...");
    check_model_dir(&yolo_dir, "whiteboard-detector", &["best.onnx", "SKILL.md"])?;
    check_model_dir(&ocr_dir, "paddleocr", &["rec.onnx", "dict.txt", "SKILL.md"])?;
    check_model_dir(&phi_dir, "cyan-lens", &["SKILL.md"])?;
    println!("✅ All models found\n");

    // Load image
    println!("📖 Loading image...");
    let image_data = fs::read(&args.image)?;
    let img = image::load_from_memory(&image_data)?;
    println!("   Size: {}x{}", img.width(), img.height());
    println!();

    // Initialize pipeline
    println!("⚙️  Initializing pipeline...");
    let start = std::time::Instant::now();
    let mut pipeline = WhiteboardPipeline::new(&yolo_dir, &ocr_dir, &phi_dir)?;
    println!("   Loaded in {:?}\n", start.elapsed());

    // Process
    println!("🚀 Processing...");
    println!("{}", "─".repeat(60));
    
    let result = pipeline.process(&image_data)?;

    // Print results
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                       RESULTS                                ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Timing
    println!("⏱️  Timing:");
    println!("   Detection:  {:>6}ms", result.timing.detection_ms);
    println!("   OCR:        {:>6}ms", result.timing.ocr_ms);
    println!("   Layout:     {:>6}ms", result.timing.layout_ms);
    println!("   Generation: {:>6}ms", result.timing.generation_ms);
    println!("   ─────────────────");
    println!("   Total:      {:>6}ms", result.timing.total_ms);
    println!();

    // Shapes
    println!("📦 Detected Shapes ({}):", result.shapes.len());
    for shape in &result.shapes {
        let text = shape.text.as_deref().unwrap_or("(no text)");
        let connections = if shape.connects_to.is_empty() {
            String::new()
        } else {
            format!(" → {:?}", shape.connects_to)
        };
        println!(
            "   [{:2}] {:20} conf={:.2}  \"{}\"{}",
            shape.id,
            shape.shape_type,
            shape.confidence,
            text,
            connections
        );
    }
    println!();

    // Diagram type
    println!("📊 Diagram Type: {:?}", result.diagram_type);
    println!();

    // Mermaid output
    println!("📝 Generated Mermaid:");
    println!("┌────────────────────────────────────────────────────────────┐");
    for line in result.mermaid.lines() {
        println!("│ {:<58} │", line);
    }
    println!("└────────────────────────────────────────────────────────────┘");
    println!();

    // Save to file
    if let Some(output_path) = args.output {
        let mut file = fs::File::create(&output_path)?;
        writeln!(file, "```mermaid")?;
        writeln!(file, "{}", result.mermaid)?;
        writeln!(file, "```")?;
        println!("💾 Saved to {:?}", output_path);
    }

    // Also output raw for piping
    if args.verbose {
        println!("\n--- RAW MERMAID (for piping) ---");
        println!("{}", result.mermaid);
    }

    Ok(())
}

fn check_model_dir(dir: &PathBuf, name: &str, required_files: &[&str]) -> Result<()> {
    if !dir.exists() {
        return Err(anyhow!(
            "Model directory not found: {:?}\n   Run: python scripts/download_models.py",
            dir
        ));
    }

    for file in required_files {
        let path = dir.join(file);
        if !path.exists() {
            return Err(anyhow!(
                "Missing {} in {}: {:?}",
                file,
                name,
                path
            ));
        }
    }

    println!("   ✓ {}", name);
    Ok(())
}

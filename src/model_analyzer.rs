use image::{DynamicImage, ImageBuffer, ImageReader, Rgb, Rgba};
use std::io::Cursor;
use xaeroid::XaeroID;
pub struct XaeroModelAnalyzer {
    pub xid: XaeroID,
}

pub fn image_bytes(file: &str) -> Vec<u8> {
    std::fs::read(file).unwrap_or_else(|e| {
        tracing::error!("failed to read image due to: {e:?}");
        vec![]
    })
}
pub fn get_pixels(bytes: Vec<u8>) -> Result<DynamicImage, Box<dyn std::error::Error>> {
    ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()?
        .decode()
        .map_err(Into::into)
}

pub fn resize_to(img: DynamicImage, width: u32, height: u32) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    use image::imageops::FilterType;
    let rgb_img = img.to_rgb8();
    let resized = image::imageops::resize(
        &rgb_img,
        width,                // YOLO input width
        height,               // YOLO input height
        FilterType::Lanczos3, // High quality resizing
    );
    tracing::info!("Resized to: 640x640 for YOLO");
    resized
}
pub fn vectorize(buffer: ImageBuffer<Rgb<u8>, Vec<u8>>) -> Vec<f32> {
    use candle_core::{Device, Shape, Tensor};

    // 1. Convert pixel values to floats and normalize
    let pixels: Vec<f32> = buffer
        .pixels() // Iterator over each pixel
        .flat_map(|pixel| {
            // Each pixel has 3 values: R, G, B
            [
                pixel[0] as f32 / 255.0, // Red: 0-255 → 0.0-1.0
                pixel[1] as f32 / 255.0, // Green: 0-255 → 0.0-1.0
                pixel[2] as f32 / 255.0, // Blue: 0-255 → 0.0-1.0
            ]
        })
        .collect();

    tracing::info!("Normalized {} pixel values", pixels.len()); // Should be 640*640*3 = 1,228,800
    pixels
}
#[cfg(test)]
mod tests {
    #[test]
    pub fn test_model_analyzer() {
        use safetensors::SafeTensors;

        let model_data_res = std::fs::read("models/yolo11n.safetensors");
        match model_data_res {
            Ok(model_data) => {
                let tensors_rs = SafeTensors::deserialize(&model_data);
                match tensors_rs {
                    Ok(tensors) => {
                        println!("YOLO11n layers found: {}", tensors.len());
                        for (name, _tensor) in tensors.tensors() {
                            println!("  {}", name);
                        }
                    }
                    Err(e) => {
                        panic!("failed to load models analyzed: {e:?}");
                    }
                }
            }
            Err(e) => {
                panic!("Error reading model data: {e:?}");
            }
        }
    }
}

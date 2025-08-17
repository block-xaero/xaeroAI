use candle_core::{Device, Tensor};
use image::{DynamicImage, ImageBuffer, ImageReader, Rgb};
use std::io::Cursor;

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

pub fn hwc_to_chw(pixels: Vec<f32>) -> Vec<f32> {
    // Current format: Height-Width-Channels (HWC)
    // [R,G,B, R,G,B, R,G,B, ...] for each row
    //
    // YOLO for example wants: Channels-Height-Width (CHW)
    // [all R values, all G values, all B values]

    let mut chw_data = vec![0.0f32; 3 * 640 * 640];
    for h in 0..640 {
        // For each row
        for w in 0..640 {
            // For each column
            let hwc_index = (h * 640 + w) * 3; // Where this pixel is in HWC format

            // Red channel: put all reds together
            chw_data[(h * 640) + w] = pixels[hwc_index];
            // Green channel: put all greens together
            chw_data[640 * 640 + h * 640 + w] = pixels[hwc_index + 1];
            // Blue channel: put all blues together
            chw_data[2 * 640 * 640 + h * 640 + w] = pixels[hwc_index + 2];
        }
    }
    chw_data
}

pub fn create_tensor(chw_data: Vec<f32>) -> Tensor {
    let input_tensor = Tensor::from_vec(
        chw_data,         // The normalized, reordered pixel data
        (1, 3, 640, 640), // Shape: [batch_size, channels, height, width]
        &Device::Cpu,     // Run on CPU (could be GPU)
    );
    match input_tensor {
        Ok(tensor) => tensor,
        Err(e) => {
            panic!("failed due to {e:?}");
        }
    }
}

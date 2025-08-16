use xaeroid::XaeroID;
pub struct XaeroModelAnalyzer {
    pub xid: XaeroID,
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

    #[test]
    pub fn test_model_quantization_level() {
        use safetensors::SafeTensors;

        let data_result = std::fs::read("models/yolo11n.safetensors");
        match data_result {
            Ok(data) => {
                let tensors_result = SafeTensors::deserialize(&data);
                match tensors_result {
                    Ok(tensors) => {
                        for (name, tensor) in tensors.tensors() {
                            println!("{}: {:?}", name, tensor.dtype());
                        }
                    }
                    Err(e) => {
                        panic!("failed to load models analyzed: {e:?}");
                    }
                }
            }
            Err(e) => {
                panic!("failed to load models analyzed: {e:?}");
            }
        }
    }
}

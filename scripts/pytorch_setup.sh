cd scripts
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install torch transformers peft datasets accelerate bitsandbytes
pip install onnx onnxruntime optimum
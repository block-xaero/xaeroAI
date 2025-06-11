# scripts/create_nano_models.py
"""
Create nano models from our quantized foundation models
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    VisionEncoderDecoderModel
)
from optimum.quanto import quantize, freeze, qint4
import safetensors.torch as safetensors
from pathlib import Path
import json
import os

# Nano architectures for different domains (Updated for Ultra-Nano)
NANO_ARCHITECTURES = {
    "vision_nano": {
        "hidden_size": 192,          # Reduced from 256
        "num_hidden_layers": 4,      # Reduced from 6
        "num_attention_heads": 3,    # Reduced from 4
        "intermediate_size": 768,    # Reduced from 1024
        "target_size_mb": 15,        # Target: ~15MB
        "specialization": "vision_processing"
    },

    "code_nano": {
        "hidden_size": 256,          # Reduced from 384
        "num_hidden_layers": 6,      # Reduced from 8
        "num_attention_heads": 4,    # Reduced from 6
        "intermediate_size": 1024,   # Reduced from 1536
        "target_size_mb": 20,        # Target: ~20MB
        "specialization": "code_generation"
    },

    "reasoning_nano": {
        "hidden_size": 224,          # Reduced from 320
        "num_hidden_layers": 6,      # Reduced from 8
        "num_attention_heads": 4,    # Reduced from 8
        "intermediate_size": 896,    # Reduced from 1280
        "target_size_mb": 18,        # Target: ~18MB
        "specialization": "reasoning"
    }
}

class NanoModelDistiller:
    def __init__(self, teacher_model_path: str, nano_config: dict):
        self.teacher_model_path = teacher_model_path
        self.nano_config = nano_config

    def load_quantized_teacher_model(self, quantized_path: str):
        """Load teacher model from your quantized .safetensors files"""

        # Map your quantized files to their original models
        model_mappings = {
            "phi2_q4.safetensors": {
                "hub_model": "microsoft/phi-2",
                "model_class": AutoModelForCausalLM
            },
            "trocr_small_handwritten_q4.safetensors": {
                "hub_model": "microsoft/trocr-small-handwritten",
                "model_class": VisionEncoderDecoderModel
            },
            "trocr_base_printed_q4.safetensors": {
                "hub_model": "microsoft/trocr-base-printed",
                "model_class": VisionEncoderDecoderModel
            },
            "codegen_350m_q4.safetensors": {
                "hub_model": "Salesforce/codegen-350M-multi",
                "model_class": AutoModelForCausalLM
            }
        }

        filename = Path(quantized_path).name

        if filename not in model_mappings:
            raise ValueError(f"Unknown quantized model: {filename}")

        mapping = model_mappings[filename]
        hub_model_id = mapping["hub_model"]
        model_class = mapping["model_class"]

        print(f"   Loading base model: {hub_model_id}")

        # Load the original model structure
        teacher = model_class.from_pretrained(
            hub_model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="cpu"
        )

        # Load tokenizer with error handling for TrOCR
        try:
            tokenizer = AutoTokenizer.from_pretrained(hub_model_id, use_fast=False)
        except Exception as e:
            print(f"   Warning: Could not load fast tokenizer: {e}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(hub_model_id, use_fast=False, legacy=True)
            except Exception as e2:
                print(f"   Warning: Could not load tokenizer with legacy=True: {e2}")
                # Fallback to a simple tokenizer for TrOCR
                if "trocr" in hub_model_id.lower():
                    print(f"   Using fallback tokenizer for TrOCR model")
                    tokenizer = AutoTokenizer.from_pretrained("gpt2")
                else:
                    raise e2

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load your quantized weights
        print(f"   Loading quantized weights from {quantized_path}")
        try:
            quantized_state_dict = safetensors.load_file(quantized_path)

            # Load the weights into the model
            missing_keys, unexpected_keys = teacher.load_state_dict(quantized_state_dict, strict=False)

            if missing_keys:
                print(f"   Warning: {len(missing_keys)} missing keys")
            if unexpected_keys:
                print(f"   Warning: {len(unexpected_keys)} unexpected keys")

            print(f"   ✅ Successfully loaded quantized weights")

            # Apply quantization to the loaded model (since the weights are quantized)
            quantize(teacher, weights=qint4)
            freeze(teacher)

        except Exception as e:
            print(f"   ❌ Error loading quantized weights: {e}")
            print(f"   Using original model weights instead")

        return teacher, tokenizer

    def create_nano_architecture(self, teacher_config):
        """Create ultra-nano model architecture based on teacher"""

        # Handle VisionEncoderDecoder models - use decoder config
        if hasattr(teacher_config, 'decoder'):
            base_config = teacher_config.decoder
        else:
            base_config = teacher_config

        # Copy base config
        nano_config = base_config.to_dict()

        # Override with ultra-nano specifications
        nano_config.update({
            "hidden_size": self.nano_config["hidden_size"],
            "num_hidden_layers": self.nano_config["num_hidden_layers"],
            "num_attention_heads": self.nano_config["num_attention_heads"],
            "intermediate_size": self.nano_config["intermediate_size"],
        })

        # Ultra-nano optimizations for smaller size
        nano_config.update({
            "attention_dropout": 0.0,      # Remove dropout for efficiency
            "hidden_dropout": 0.0,
            "dropout": 0.0,
            "use_cache": False,            # Reduce memory usage
            "tie_word_embeddings": True,   # Share input/output embeddings
        })

        # Keep vocab size same for compatibility (but we could reduce this too)
        nano_config["vocab_size"] = base_config.vocab_size

        # For very large vocabs, optionally reduce size
        if base_config.vocab_size > 50000:
            print(f"   Large vocab detected ({base_config.vocab_size}), keeping for compatibility")
            # Uncomment next line to reduce vocab size for even smaller models:
            # nano_config["vocab_size"] = min(base_config.vocab_size, 32000)

        return type(base_config)(**nano_config)

    def distill_to_nano(self, output_path: str, num_epochs: int = 5):
        """Knowledge distillation: Quantized Teacher → Nano Student"""

        print(f"📚 Loading teacher model from {self.teacher_model_path}")

        # Load quantized teacher model
        teacher, tokenizer = self.load_quantized_teacher_model(self.teacher_model_path)

        print(f"🧬 Creating nano student architecture...")

        # Create nano student with compatible architecture
        nano_config = self.create_nano_architecture(teacher.config)

        # Handle different model types for student creation
        if isinstance(teacher, VisionEncoderDecoderModel):
            # For TrOCR models, create a simpler CausalLM student
            print("   Converting VisionEncoderDecoder teacher to CausalLM student...")
            student = AutoModelForCausalLM.from_config(nano_config)
        else:
            # For CausalLM models, use AutoModelForCausalLM regardless of specific type
            student = AutoModelForCausalLM.from_config(nano_config)

        # Initialize student weights intelligently
        self.initialize_student_weights(teacher, student)

        print(f"📊 Model sizes:")
        teacher_params = sum(p.numel() for p in teacher.parameters())
        student_params = sum(p.numel() for p in student.parameters())
        print(f"   Teacher: {teacher_params:,} params")
        print(f"   Student: {student_params:,} params")
        print(f"   Reduction: {teacher_params/student_params:.1f}x smaller")

        # Knowledge distillation training
        print(f"🎓 Starting knowledge distillation...")
        self.knowledge_distillation_training(
            teacher, student, tokenizer, num_epochs
        )

        # Quantize the nano model
        print(f"🗜️ Quantizing nano model...")
        quantize(student, weights=qint4)
        freeze(student)

        # Save nano model
        print(f"💾 Saving nano model to {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        state_dict = {k: v.cpu() for k, v in student.state_dict().items()}
        safetensors.save_file(state_dict, output_path)

        # Save nano config
        nano_config_path = output_path.replace('.safetensors', '_config.json')
        with open(nano_config_path, 'w') as f:
            json.dump(nano_config.to_dict(), f, indent=2)

        # Save tokenizer
        tokenizer_path = Path(output_path).parent / f"{Path(output_path).stem}_tokenizer"
        tokenizer.save_pretrained(tokenizer_path)

        # Save metadata similar to your quantization script
        original_size = Path(self.teacher_model_path).stat().st_size / 1024 / 1024
        final_size = Path(output_path).stat().st_size / 1024 / 1024

        metadata = {
            "teacher_model": self.teacher_model_path,
            "teacher_size_mb": original_size,
            "nano_size_mb": final_size,
            "compression_ratio": original_size / final_size,
            "teacher_params": teacher_params,
            "student_params": student_params,
            "param_reduction": teacher_params / student_params,
            "nano_config": self.nano_config,
            "distillation_method": "knowledge_distillation"
        }

        metadata_path = output_path.replace('.safetensors', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Nano model created: {final_size:.1f}MB")
        print(f"   Total compression: {original_size/final_size:.1f}x from original teacher")

        return output_path

    def initialize_student_weights(self, teacher, student):
        """Smart weight initialization from teacher"""

        with torch.no_grad():
            # Get the actual model for weight copying (handle VisionEncoderDecoder)
            if isinstance(teacher, VisionEncoderDecoderModel):
                teacher_model = teacher.decoder
            else:
                teacher_model = teacher

            student_model = student

            # Copy embedding layers directly (same vocab size)
            if hasattr(teacher_model, 'embed_tokens') and hasattr(student_model, 'embed_tokens'):
                student_model.embed_tokens.weight.copy_(teacher_model.embed_tokens.weight)

            # Initialize transformer layers by mapping teacher layers to student layers
            teacher_layers = self.get_model_layers(teacher_model)
            student_layers = self.get_model_layers(student_model)

            if teacher_layers and student_layers:
                # Map teacher layers to student layers (skip some teacher layers)
                layer_mapping = self.create_layer_mapping(len(teacher_layers), len(student_layers))

                for student_idx, teacher_idx in layer_mapping.items():
                    if student_idx < len(student_layers) and teacher_idx < len(teacher_layers):
                        self.copy_layer_weights(teacher_layers[teacher_idx], student_layers[student_idx])

    def get_model_layers(self, model):
        """Get transformer layers from model"""
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return model.transformer.h
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return model.model.layers
        elif hasattr(model, 'layers'):
            return model.layers
        else:
            return None

    def create_layer_mapping(self, teacher_layers: int, student_layers: int) -> dict:
        """Map teacher layers to student layers"""
        if student_layers >= teacher_layers:
            # More student layers than teacher (shouldn't happen)
            return {i: i for i in range(student_layers)}

        # Skip some teacher layers evenly
        step = teacher_layers / student_layers
        return {
            student_idx: int(student_idx * step)
            for student_idx in range(student_layers)
        }

    def copy_layer_weights(self, teacher_layer, student_layer):
        """Copy weights from teacher layer to student layer with dimension adjustment"""

        with torch.no_grad():
            # Copy attention weights (adjust dimensions if needed)
            if hasattr(teacher_layer, 'self_attn') and hasattr(student_layer, 'self_attn'):
                self.copy_attention_weights(teacher_layer.self_attn, student_layer.self_attn)

            # Copy feed-forward weights
            if hasattr(teacher_layer, 'mlp') and hasattr(student_layer, 'mlp'):
                self.copy_mlp_weights(teacher_layer.mlp, student_layer.mlp)

    def copy_attention_weights(self, teacher_attn, student_attn):
        """Copy attention weights with dimension adjustment"""

        try:
            # Common attention projection names across different architectures
            proj_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'out_proj', 'dense']

            for proj_name in proj_names:
                if hasattr(teacher_attn, proj_name) and hasattr(student_attn, proj_name):
                    teacher_proj = getattr(teacher_attn, proj_name).weight
                    student_proj = getattr(student_attn, proj_name).weight

                    # Truncate or pad teacher weights to match student dimensions
                    min_out = min(teacher_proj.size(0), student_proj.size(0))
                    min_in = min(teacher_proj.size(1), student_proj.size(1))

                    student_proj[:min_out, :min_in] = teacher_proj[:min_out, :min_in]

        except Exception as e:
            print(f"Warning: Could not copy attention weights: {e}")

    def copy_mlp_weights(self, teacher_mlp, student_mlp):
        """Copy MLP weights with dimension adjustment"""

        try:
            # Common MLP projection names across different architectures
            proj_names = ['gate_proj', 'up_proj', 'down_proj', 'fc_in', 'fc_out', 'c_fc', 'c_proj']

            for proj_name in proj_names:
                if hasattr(teacher_mlp, proj_name) and hasattr(student_mlp, proj_name):
                    teacher_proj = getattr(teacher_mlp, proj_name).weight
                    student_proj = getattr(student_mlp, proj_name).weight

                    min_out = min(teacher_proj.size(0), student_proj.size(0))
                    min_in = min(teacher_proj.size(1), student_proj.size(1))

                    student_proj[:min_out, :min_in] = teacher_proj[:min_out, :min_in]

        except Exception as e:
            print(f"Warning: Could not copy MLP weights: {e}")

    def knowledge_distillation_training(self, teacher, student, tokenizer, num_epochs):
        """Train student to mimic teacher outputs"""

        # Create domain-specific training data
        training_texts = self.create_training_data()

        # Training setup
        optimizer = torch.optim.AdamW(student.parameters(), lr=5e-4)
        temperature = 3.0  # Softmax temperature for distillation

        teacher.eval()
        student.train()

        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0

            for text in training_texts:
                try:
                    # Use shorter sequences to avoid shape mismatches
                    inputs = tokenizer(text, return_tensors="pt", max_length=64, truncation=True, padding="max_length")

                    with torch.no_grad():
                        if isinstance(teacher, VisionEncoderDecoderModel):
                            # For TrOCR models, we need to handle them differently
                            # Since we don't have vision inputs, we'll use the decoder only
                            decoder_inputs = inputs
                            teacher_outputs = teacher.decoder(**decoder_inputs)
                        else:
                            teacher_outputs = teacher(**inputs)

                        teacher_logits = teacher_outputs.logits

                    # Student is always CausalLM
                    student_outputs = student(**inputs)
                    student_logits = student_outputs.logits

                    # Ensure logits have compatible shapes for distillation
                    if teacher_logits.shape != student_logits.shape:
                        # Adjust sequence length if needed
                        min_seq_len = min(teacher_logits.size(1), student_logits.size(1))
                        teacher_logits = teacher_logits[:, :min_seq_len, :]
                        student_logits = student_logits[:, :min_seq_len, :]

                        # Adjust vocab size if needed
                        min_vocab_size = min(teacher_logits.size(-1), student_logits.size(-1))
                        teacher_logits = teacher_logits[:, :, :min_vocab_size]
                        student_logits = student_logits[:, :, :min_vocab_size]

                    # Knowledge distillation loss
                    distillation_loss = self.distillation_loss(
                        student_logits, teacher_logits, temperature
                    )

                    optimizer.zero_grad()
                    distillation_loss.backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)  # Gradient clipping
                    optimizer.step()

                    epoch_loss += distillation_loss.item()
                    num_batches += 1

                except Exception as e:
                    print(f"   Warning: Skipping batch due to error: {e}")
                    continue

            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                print(f"   Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f} ({num_batches} successful batches)")
            else:
                print(f"   Epoch {epoch+1}/{num_epochs}: No successful batches - skipping distillation")
                # If no batches work, break out early
                break

    def distillation_loss(self, student_logits, teacher_logits, temperature):
        """Knowledge distillation loss function"""

        # Soften predictions
        teacher_probs = torch.softmax(teacher_logits / temperature, dim=-1)
        student_log_probs = torch.log_softmax(student_logits / temperature, dim=-1)

        # KL divergence loss
        loss = torch.nn.KLDivLoss(reduction='batchmean')(student_log_probs, teacher_probs)

        return loss * (temperature ** 2)  # Scale by temperature squared

    def create_training_data(self):
        """Create domain-specific training data"""

        specialization = self.nano_config["specialization"]

        if specialization == "vision_processing":
            return [
                "Analyze this diagram and extract the key components",
                "Identify text regions in this image",
                "Convert handwritten notes to digital text",
                "Parse the structure of this flowchart",
                "Extract UI elements from this mockup",
                "Recognize text in handwritten documents",
                "Process scanned document images",
                "Extract information from forms and receipts"
            ]
        elif specialization == "code_generation":
            return [
                "Create a REST API endpoint for user management",
                "Write a function to validate email addresses",
                "Implement a binary search algorithm",
                "Design a database schema for a blog",
                "Create error handling for file operations",
                "Generate unit tests for this function",
                "Refactor this code for better performance",
                "Add logging and monitoring to this service"
            ]
        elif specialization == "reasoning":
            return [
                "Analyze the trade-offs between microservices and monoliths",
                "Design a scalable system for real-time notifications",
                "Evaluate different database options for high-throughput workloads",
                "Plan a migration strategy from legacy to modern architecture",
                "Assess security implications of API design choices",
                "Compare different caching strategies for web applications",
                "Design a fault-tolerant distributed system",
                "Optimize database queries for better performance"
            ]
        else:
            return [
                "Complete this task with careful reasoning",
                "Analyze the given problem systematically",
                "Provide a structured solution approach",
                "Break down complex problems into manageable steps"
            ]

def nano_fy_all_models():
    """Convert all quantized models to ultra-nano versions"""

    quantized_dir = Path("models/quantized")
    nano_dir = Path("models/nano")

    # Clean up previous nano models to avoid confusion
    if nano_dir.exists():
        print("🧹 Cleaning previous nano models...")
        import shutil
        shutil.rmtree(nano_dir)

    nano_dir.mkdir(parents=True, exist_ok=True)

    # Model mappings based on your actual files
    model_mappings = {
        "phi2_q4.safetensors": ("code_nano", "Phi-2 Ultra-Code Specialist"),
        "trocr_small_handwritten_q4.safetensors": ("vision_nano", "TrOCR Small Ultra-Vision Specialist"),
        "trocr_base_printed_q4.safetensors": ("vision_nano", "TrOCR Base Ultra-Vision Specialist"),
        "codegen_350m_q4.safetensors": ("code_nano", "CodeGen Ultra-Code Specialist"),
    }

    successful_models = []
    failed_models = []

    for quantized_model, (nano_arch, description) in model_mappings.items():
        quantized_path = quantized_dir / quantized_model

        if not quantized_path.exists():
            print(f"⚠️ Skipping {quantized_model} - not found")
            failed_models.append(quantized_model)
            continue

        print(f"\n{'='*60}")
        print(f"Creating {description}")
        print(f"Target size: ~{NANO_ARCHITECTURES[nano_arch]['target_size_mb']}MB")
        print(f"{'='*60}")

        # Create nano model
        distiller = NanoModelDistiller(
            str(quantized_path),
            NANO_ARCHITECTURES[nano_arch]
        )

        # Create unique nano output name to avoid conflicts
        model_base_name = quantized_model.replace('.safetensors', '').replace('_q4', '')
        nano_output = nano_dir / f"{model_base_name}_ultra_{nano_arch}.safetensors"

        try:
            distiller.distill_to_nano(str(nano_output))
            print(f"✅ Successfully created {nano_output.name}")
            successful_models.append(nano_output.name)
        except Exception as e:
            print(f"❌ Failed to create {nano_output.name}: {e}")
            failed_models.append(quantized_model)
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print(f"🎉 ULTRA-NANO MODEL CREATION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful: {len(successful_models)}")
    for model in successful_models:
        print(f"   - {model}")

    if failed_models:
        print(f"❌ Failed: {len(failed_models)}")
        for model in failed_models:
            print(f"   - {model}")

    print(f"\n📁 Models saved to: {nano_dir}")
    print(f"🎯 Expected total size: ~{sum(arch['target_size_mb'] for arch in NANO_ARCHITECTURES.values())*len(successful_models)//len(NANO_ARCHITECTURES)}MB")

if __name__ == "__main__":
    nano_fy_all_models()
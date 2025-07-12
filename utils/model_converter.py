import torch
import os
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Dict, Any

class ModelConverter:
    """Core functionality for model conversion operations"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def convert_to_torchserve(self, vae_path: str, diffuser_path: str, output_dir: str) -> Dict[str, Any]:
        """Convert PyTorch models to ONNX format for TorchServe deployment"""
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            onnx_dir = os.path.join(output_dir, "onnx_models")
            os.makedirs(onnx_dir, exist_ok=True)
            
            print("Starting model conversion to ONNX format...")
            
            # 1. Export CLIP Text Encoder
            print("Converting CLIP Text Encoder...")
            self._export_clip_text_encoder(onnx_dir)
            
            # 2. Export VAE Decoder
            print("Converting VAE Decoder...")
            self._export_vae_decoder(vae_path, onnx_dir)
            
            # 3. Export UNet
            print("Converting UNet...")
            self._export_unet(diffuser_path, onnx_dir)
            
            # Verify all models were created
            expected_models = [
                "clip_text_encoder.onnx",
                "vae_decoder.onnx",
                "unet.onnx"
            ]
            
            created_models = []
            for model_name in expected_models:
                model_path = os.path.join(onnx_dir, model_name)
                if os.path.exists(model_path):
                    created_models.append(model_name)
                    print(f"✅ {model_name} created successfully")
                else:
                    print(f"❌ {model_name} creation failed")
            
            return {
                "conversion_status": "completed",
                "output_dir": onnx_dir,
                "created_models": created_models,
                "total_models": len(created_models),
                "model_paths": [os.path.join(onnx_dir, model) for model in created_models]
            }
            
        except Exception as e:
            raise Exception(f"Model conversion failed: {e}")
    
    def _export_clip_text_encoder(self, output_dir: str):
        """Export CLIP text encoder to ONNX"""
        model_name = "openai/clip-vit-large-patch14"
        clip_model = CLIPTextModel.from_pretrained(model_name).to(self.device).eval()
        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        
        # Create dummy input
        dummy_input = tokenizer(
            "This is a dummy prompt", 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=77
        )
        input_ids = dummy_input["input_ids"].to(self.device)
        attention_mask = dummy_input["attention_mask"].to(self.device)
        
        # Export to ONNX
        torch.onnx.export(
            clip_model,
            (input_ids, attention_mask),
            os.path.join(output_dir, "clip_text_encoder.onnx"),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch_size"}, 
                "attention_mask": {0: "batch_size"}, 
                "last_hidden_state": {0: "batch_size"}
            },
            opset_version=17,
            do_constant_folding=True
        )
    
    def _export_vae_decoder(self, vae_path: str, output_dir: str):
        """Export VAE decoder to ONNX"""
        
        class VAEDecoderWrapper(torch.nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.vae = vae
            
            def forward(self, latents):
                return self.vae.decode(latents).sample
        
        # Load VAE model
        vae = torch.load(vae_path, map_location=self.device, weights_only=False).eval()
        vae_decoder = VAEDecoderWrapper(vae)
        
        # Create dummy input
        dummy_latents = torch.randn(1, 4, 32, 32).to(self.device)
        
        # Export to ONNX
        torch.onnx.export(
            vae_decoder,
            dummy_latents,
            os.path.join(output_dir, "vae_decoder.onnx"),
            input_names=["latents"],
            output_names=["image"],
            dynamic_axes={"latents": {0: "batch_size"}, "image": {0: "batch_size"}},
            opset_version=17,
            do_constant_folding=True
        )
    
    def _export_unet(self, diffuser_path: str, output_dir: str):
        """Export UNet to ONNX"""
        
        class UNetWrapper(torch.nn.Module):
            def __init__(self, unet):
                super().__init__()
                self.unet = unet
            
            def forward(self, latents, timesteps, encoder_hidden_states):
                return self.unet(latents, timesteps, encoder_hidden_states).sample
        
        # Load UNet model
        diffuser = torch.load(diffuser_path, map_location=self.device, weights_only=False).eval()
        unet_wrapper = UNetWrapper(diffuser)
        
        # Create dummy inputs
        dummy_latents = torch.randn(1, 4, 32, 32).to(self.device)
        dummy_timestep = torch.tensor([50], dtype=torch.float32).to(self.device)
        dummy_text_emb = torch.randn(1, 77, 768).to(self.device)
        
        # Export to ONNX
        torch.onnx.export(
            unet_wrapper,
            (dummy_latents, dummy_timestep, dummy_text_emb),
            os.path.join(output_dir, "unet.onnx"),
            input_names=["latents", "timesteps", "encoder_hidden_states"],
            output_names=["noise_pred"],
            dynamic_axes={
                "latents": {0: "batch_size"}, 
                "encoder_hidden_states": {0: "batch_size"}, 
                "noise_pred": {0: "batch_size"}
            },
            opset_version=17,
            do_constant_folding=True
        )

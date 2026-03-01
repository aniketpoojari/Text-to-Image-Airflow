import torch
import os
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from typing import Dict, Any, Tuple


class ModelConverter:
    """Core functionality for model conversion operations"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def convert_to_torchserve(
        self,
        diffuser_path: str,
        output_dir: str,
        image_size: Tuple[int, int] = (128, 128),
    ) -> Dict[str, Any]:
        """Convert PyTorch models to ONNX format for TorchServe deployment.

        The VAE is loaded from the pretrained HuggingFace checkpoint
        (stabilityai/sd-vae-ft-mse) rather than from a local file, since the
        VAE is frozen during training and never fine-tuned.
        """
        os.makedirs(output_dir, exist_ok=True)
        onnx_dir = os.path.join(output_dir, "onnx_models")
        os.makedirs(onnx_dir, exist_ok=True)

        latent_h = image_size[0] // 8
        latent_w = image_size[1] // 8

        print("Starting model conversion to ONNX format...")
        print(f"Image size: {image_size}, Latent size: ({latent_h}, {latent_w})")

        self._export_clip_text_encoder(onnx_dir)
        self._export_vae_decoder(onnx_dir, latent_h, latent_w)
        self._export_unet(diffuser_path, onnx_dir, latent_h, latent_w)

        expected_models = ["clip_text_encoder.onnx", "vae_decoder.onnx", "unet.onnx"]
        created_models = []
        for model_name in expected_models:
            model_path = os.path.join(onnx_dir, model_name)
            if os.path.exists(model_path):
                created_models.append(model_name)
                print(f"  [ok] {model_name}")
            else:
                print(f"  [MISSING] {model_name}")

        return {
            "conversion_status": "completed",
            "output_dir": onnx_dir,
            "created_models": created_models,
            "total_models": len(created_models),
            "model_paths": [os.path.join(onnx_dir, m) for m in created_models],
        }

    def _export_clip_text_encoder(self, output_dir: str):
        """Export CLIP text encoder to ONNX"""
        model_name = "openai/clip-vit-large-patch14"
        clip_model = CLIPTextModel.from_pretrained(model_name).to(self.device).eval()
        tokenizer = CLIPTokenizer.from_pretrained(model_name)

        dummy_input = tokenizer(
            "This is a dummy prompt",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        )
        input_ids = dummy_input["input_ids"].to(self.device)
        attention_mask = dummy_input["attention_mask"].to(self.device)

        torch.onnx.export(
            clip_model,
            (input_ids, attention_mask),
            os.path.join(output_dir, "clip_text_encoder.onnx"),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "last_hidden_state": {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
        )

    def _export_vae_decoder(self, output_dir: str, latent_h: int, latent_w: int):
        """Export VAE decoder to ONNX from the pretrained HuggingFace checkpoint"""

        class VAEDecoderWrapper(torch.nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.vae = vae

            def forward(self, latents):
                return self.vae.decode(latents).sample

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(self.device).eval()
        vae_decoder = VAEDecoderWrapper(vae)

        dummy_latents = torch.randn(1, 4, latent_h, latent_w).to(self.device)

        torch.onnx.export(
            vae_decoder,
            dummy_latents,
            os.path.join(output_dir, "vae_decoder.onnx"),
            input_names=["latents"],
            output_names=["image"],
            dynamic_axes={"latents": {0: "batch_size"}, "image": {0: "batch_size"}},
            opset_version=17,
            do_constant_folding=True,
        )

    def _export_unet(self, diffuser_path: str, output_dir: str, latent_h: int, latent_w: int):
        """Export UNet to ONNX.

        Uses batch_size=2 for dummy inputs to support CFG inference, which
        concatenates conditional and unconditional inputs into a single forward
        pass.  The timesteps axis is also marked dynamic so the ONNX model
        accepts any batch size at inference time.
        """

        class UNetWrapper(torch.nn.Module):
            def __init__(self, unet):
                super().__init__()
                self.unet = unet

            def forward(self, latents, timesteps, encoder_hidden_states):
                return self.unet(latents, timesteps, encoder_hidden_states).sample

        diffuser = torch.load(diffuser_path, map_location=self.device, weights_only=False).eval()
        unet_wrapper = UNetWrapper(diffuser)

        # batch_size=2 for CFG (conditional + unconditional in one forward pass)
        dummy_latents = torch.randn(2, 4, latent_h, latent_w).to(self.device)
        dummy_timestep = torch.tensor([50, 50], dtype=torch.float32).to(self.device)
        dummy_text_emb = torch.randn(2, 77, 768).to(self.device)

        torch.onnx.export(
            unet_wrapper,
            (dummy_latents, dummy_timestep, dummy_text_emb),
            os.path.join(output_dir, "unet.onnx"),
            input_names=["latents", "timesteps", "encoder_hidden_states"],
            output_names=["noise_pred"],
            dynamic_axes={
                "latents": {0: "batch_size"},
                "timesteps": {0: "batch_size"},
                "encoder_hidden_states": {0: "batch_size"},
                "noise_pred": {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
        )

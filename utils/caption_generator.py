import torch
import os
from PIL import Image
from tqdm import tqdm
from typing import Dict, Any
from orchestrators.config_models import PipelineConfig


class CaptionGenerator:
    """Generate image captions using Microsoft Florence-2.

    Runs locally (before data is uploaded to S3) so the pipeline can work on
    any raw image dataset without pre-existing captions.
    """

    def generate_captions(self, config: PipelineConfig) -> Dict[str, Any]:
        """Generate captions for all images that don't yet have one."""
        from transformers import AutoProcessor, AutoModelForCausalLM

        data_path = config.caption_generator.data_path
        model_name = config.caption_generator.model_name
        batch_size = config.caption_generator.batch_size

        image_dir = os.path.join(data_path, "images")
        captions_dir = os.path.join(data_path, "captions")

        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        os.makedirs(captions_dir, exist_ok=True)

        # Find images that still need captions
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        all_images = [f for f in os.listdir(image_dir) if f.lower().endswith(image_extensions)]

        already_done = {os.path.splitext(f)[0] for f in os.listdir(captions_dir) if f.endswith('.txt')}
        remaining = [(f, os.path.join(image_dir, f)) for f in all_images
                     if os.path.splitext(f)[0] not in already_done]

        if not remaining:
            print("All captions already generated. Skipping.")
            return {
                "caption_status": "skipped",
                "total_images": len(all_images),
                "generated": 0,
            }

        print(f"Found {len(remaining)} images needing captions (out of {len(all_images)} total).")
        print(f"Loading {model_name}...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        print(f"Model loaded on {device.upper()}.")

        filenames = [item[0] for item in remaining]
        paths = [item[1] for item in remaining]
        generated_count = 0

        for i in tqdm(range(0, len(paths), batch_size), desc="Captioning"):
            batch_paths = paths[i:i + batch_size]
            batch_files = filenames[i:i + batch_size]

            try:
                captions = self._process_batch(batch_paths, processor, model, device)
                for filename, caption in zip(batch_files, captions):
                    caption_file = os.path.join(captions_dir, os.path.splitext(filename)[0] + ".txt")
                    with open(caption_file, 'w', encoding='utf-8') as f:
                        f.write(caption.strip())
                generated_count += len(batch_files)

            except torch.cuda.OutOfMemoryError:
                print(f"OOM at batch {i // batch_size + 1}. Falling back to single-image mode...")
                for path, filename in zip(batch_paths, batch_files):
                    try:
                        torch.cuda.empty_cache()
                        captions = self._process_batch([path], processor, model, device)
                        caption_file = os.path.join(captions_dir, os.path.splitext(filename)[0] + ".txt")
                        with open(caption_file, 'w', encoding='utf-8') as f:
                            f.write(captions[0].strip())
                        generated_count += 1
                    except Exception as e:
                        print(f"  Failed on {filename}: {e}")

        return {
            "caption_status": "completed",
            "total_images": len(all_images),
            "generated": generated_count,
            "captions_dir": captions_dir,
        }

    def _process_batch(self, image_paths, processor, model, device):
        """Run Florence-2 on a batch of images and return caption strings."""
        images = [Image.open(p).convert('RGB').resize((512, 512), Image.Resampling.LANCZOS)
                  for p in image_paths]
        prompts = ["<MORE_DETAILED_CAPTION>"] * len(images)

        dtype = torch.float16 if device == "cuda" else torch.float32
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device, dtype)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=150,
                num_beams=3,
                do_sample=False,
            )

        captions = []
        for i, generated_text in enumerate(processor.batch_decode(generated_ids, skip_special_tokens=True)):
            parsed = processor.post_process_generation(
                generated_text,
                task="<MORE_DETAILED_CAPTION>",
                image_size=(images[i].width, images[i].height),
            )
            captions.append(parsed["<MORE_DETAILED_CAPTION>"])

        return captions

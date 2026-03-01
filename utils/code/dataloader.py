import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
from typing import Tuple
import os


class TextImageDataLoader(Dataset):
    """Text-image dataset with pre-computed CLIP embeddings.

    Text embeddings are computed once at initialisation so that CLIP never
    runs inside the training loop.  The text encoder is released from GPU
    memory after pre-computation so that DataLoader worker processes can be
    spawned safely without CUDA conflicts.

    Images are normalised to [-1, 1] to match the expected input range of
    the pretrained VAE (stabilityai/sd-vae-ft-mse).
    """

    def __init__(
        self,
        datadir: str,
        range: Tuple[int, int],
        image_size: Tuple[int, int],
        max_text_length: int,
        device: str = "cuda",
        training: bool = True,
    ):
        super().__init__()

        self.datadir = datadir
        self.datalist = os.listdir(datadir + '/images')
        self.datalist = self.datalist[range[0]:range[1]]
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.device = device

        model_name = "openai/clip-vit-large-patch14"
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        text_encoder = CLIPTextModel.from_pretrained(model_name).to(device).eval()

        # Training transform includes random flip augmentation.
        # Validation transform is deterministic — no augmentation.
        base_transforms = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
        if training:
            base_transforms.insert(1, transforms.RandomHorizontalFlip(p=0.5))
        self.image_transform = transforms.Compose(base_transforms)

        self._precompute_embeddings(text_encoder)
        self._compute_null_embedding(text_encoder)

        # Release GPU memory — workers must not access CUDA objects
        del text_encoder
        torch.cuda.empty_cache()

    def _precompute_embeddings(self, text_encoder: CLIPTextModel):
        """Encode every caption once and cache the result on CPU."""
        self.embeddings_cache = []
        for image_file in self.datalist:
            caption_file = image_file.replace('.jpg', '.txt')
            caption_path = os.path.join(self.datadir, 'captions', caption_file)
            with open(caption_path, 'r') as f:
                text = f.readline().strip()

            tokens = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                output = text_encoder(
                    input_ids=tokens.input_ids.to(self.device),
                    attention_mask=tokens.attention_mask.to(self.device),
                )
            self.embeddings_cache.append(output.last_hidden_state.squeeze(0).cpu())

    def _compute_null_embedding(self, text_encoder: CLIPTextModel):
        """Pre-compute the null (empty-string) embedding used for CFG."""
        null_tokens = self.tokenizer(
            "",
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = text_encoder(
                input_ids=null_tokens.input_ids.to(self.device),
                attention_mask=null_tokens.attention_mask.to(self.device),
            )
        self.null_text_embedding = output.last_hidden_state.squeeze(0).cpu()

    def get_null_embedding(self, batch_size: int) -> torch.Tensor:
        """Return the null embedding broadcast to the requested batch size."""
        return self.null_text_embedding.unsqueeze(0).expand(batch_size, -1, -1)

    def __len__(self) -> int:
        return len(self.datalist)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_file = self.datalist[idx]
        image = Image.open(os.path.join(self.datadir, 'images', image_file)).convert("RGB")
        image = self.image_transform(image)
        return image, self.embeddings_cache[idx]

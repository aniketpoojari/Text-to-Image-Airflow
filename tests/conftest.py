"""
Shared pytest fixtures for the Text-to-Image-Airflow test suite.

All fixtures are designed to run without GPU, AWS credentials, or
large model downloads.
"""
import sys
from unittest.mock import MagicMock

# ── Pre-mock packages unavailable in CI ───────────────────────────────────────
# Must happen BEFORE any test module imports orchestrators or utils that
# transitively import these packages.
for _pkg in ('sagemaker', 'sagemaker.pytorch', 'sagemaker.session',
             'sagemaker.inputs', 'sagemaker.estimator'):
    sys.modules.setdefault(_pkg, MagicMock())

import os
import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from unittest.mock import patch

# Make sure project root and orchestrators are importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "orchestrators"))
sys.path.insert(0, str(PROJECT_ROOT / "utils"))


# ── Minimal valid config dict (mirrors pipeline_config.yaml structure) ─────

VALID_CONFIG_DICT = {
    "caption_generator": {
        "model_name": "microsoft/Florence-2-large",
        "batch_size": 10,
        "data_path": "data/raw/flowers",
    },
    "data": {
        "train_size": 30,
        "val_size": 3,
        "raw_data_path": "data/raw/flowers.zip",
    },
    "vae": {"image_size": "128,128"},
    "unet": {
        "image_size": "16,16",
        "in_channels": 4,
        "out_channels": 4,
        "down_block_types": "CrossAttnDownBlock2D,DownBlock2D,CrossAttnDownBlock2D",
        "up_block_types": "CrossAttnUpBlock2D,UpBlock2D,CrossAttnUpBlock2D",
        "mid_block_type": "UNetMidBlock2DCrossAttn",
        "block_out_channels": "64,128,256",
        "layers_per_block": 2,
        "norm_num_groups": 32,
        "cross_attention_dim": 512,
        "attention_head_dim": 8,
        "dropout": 0.1,
        "time_embedding_type": "positional",
        "act_fn": "silu",
        "learning_rate": 1e-3,
    },
    "ddpm_scheduler": {"T": 1000, "beta_schedule": "squaredcos_cap_v2"},
    "clip": {"max_length": 77},
    "training": {
        "batch_size": 4,
        "weight_decay": 1e-4,
        "num_epochs": 1,
        "cfg_dropout_prob": 0.1,
    },
    "mlflow": {
        "experiment_name": "test-experiment",
        "run_name": "test-run",
        "registered_model_name": "test-model",
        "server_uri": "http://localhost:5000",
        "s3_mlruns_bucket": "test-bucket",
        "tracking_username": "user",
        "tracking_password": "pass",
    },
    "sagemaker": {
        "role": "arn:aws:iam::123456789:role/SageMakerRole",
        "instance_count": 1,
        "instance_type": "ml.g4dn.xlarge",
        "framework_version": "2.0.0",
        "py_version": "py310",
        "max_wait": 7200,
        "max_run": 7200,
        "use_spot_instances": True,
        "s3_train_data": "s3://test-bucket/",
        "entry_point": "training_sagemaker_deepspeed.py",
        "source_dir": "utils/code",
    },
}


@pytest.fixture
def valid_config_dict():
    """Return a copy of the valid config dict."""
    import copy
    return copy.deepcopy(VALID_CONFIG_DICT)


@pytest.fixture
def mock_pipeline_config(valid_config_dict):
    """Return a validated PipelineConfig built from the in-memory dict."""
    from config_models import PipelineConfig
    return PipelineConfig(**valid_config_dict)


@pytest.fixture
def tmp_flower_dir(tmp_path):
    """
    Create a minimal flowers dataset directory structure:
      <tmp>/flowers/images/image_0000N.jpg   (5 tiny 32x32 images)
      <tmp>/flowers/captions/image_0000N.txt (matching captions)
    Returns the path to the flowers directory.
    """
    flowers_dir = tmp_path / "flowers"
    images_dir = flowers_dir / "images"
    captions_dir = flowers_dir / "captions"
    images_dir.mkdir(parents=True)
    captions_dir.mkdir(parents=True)

    for i in range(1, 6):
        name = f"image_{i:05d}"
        # Tiny solid-colour RGB image
        img = Image.fromarray(
            np.full((32, 32, 3), fill_value=(i * 40, 100, 200), dtype=np.uint8)
        )
        img.save(images_dir / f"{name}.jpg")
        (captions_dir / f"{name}.txt").write_text(f"A beautiful flower number {i}")

    return str(flowers_dir)


# ── CLIP stub helpers ────────────────────────────────────────────────────────

def make_clip_tokenizer_mock():
    """Return a mock CLIPTokenizer that produces fixed-length token tensors."""
    mock = MagicMock()
    def _tokenize(text, padding=None, truncation=None, max_length=77, return_tensors="pt"):
        ids = torch.zeros(1, max_length, dtype=torch.long)
        mask = torch.ones(1, max_length, dtype=torch.long)
        result = MagicMock()
        result.input_ids = ids
        result.attention_mask = mask
        return result
    mock.return_value = _tokenize("dummy")
    mock.side_effect = None
    mock.__call__ = _tokenize
    # Make it callable directly
    tokenizer = MagicMock(side_effect=_tokenize)
    tokenizer.from_pretrained = MagicMock(return_value=tokenizer)
    return tokenizer


def make_clip_text_encoder_mock(max_length=77, hidden_dim=768):
    """Return a mock CLIPTextModel that outputs zero hidden states."""
    encoder = MagicMock()

    class FakeOutput:
        last_hidden_state = torch.zeros(1, max_length, hidden_dim)

    encoder.return_value = FakeOutput()
    encoder.to = MagicMock(return_value=encoder)
    encoder.eval = MagicMock(return_value=encoder)
    encoder.parameters = MagicMock(return_value=iter([]))
    encoder.from_pretrained = MagicMock(return_value=encoder)
    return encoder

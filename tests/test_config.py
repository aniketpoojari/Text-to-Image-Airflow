"""
Tests for Pydantic config model validation.
No GPU or AWS required.
"""
import copy
import pytest
import yaml
from pathlib import Path
from pydantic import ValidationError

from config_models import (
    PipelineConfig,
    TaskResult,
    CaptionConfig,
    DataConfig,
    VAEConfig,
    UNetConfig,
    DDPMSchedulerConfig,
    TrainingConfig,
)


CONFIG_PATH = Path(__file__).parent.parent / "config" / "pipeline_config.yaml"


# ── pipeline_config.yaml round-trip ─────────────────────────────────────────

def test_pipeline_config_loads_from_yaml():
    """The checked-in YAML must deserialise into PipelineConfig without errors."""
    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    config = PipelineConfig(**raw)
    assert config.ddpm_scheduler.T == 1000
    assert config.ddpm_scheduler.beta_schedule == "squaredcos_cap_v2"
    assert config.training.cfg_dropout_prob == 0.1


def test_pipeline_config_from_dict(valid_config_dict):
    config = PipelineConfig(**valid_config_dict)
    assert config.data.train_size == 30
    assert config.data.val_size == 3
    assert config.unet.attention_head_dim == 8


# ── VAEConfig validator ──────────────────────────────────────────────────────

def test_vae_image_size_valid():
    cfg = VAEConfig(image_size="128,128")
    assert cfg.image_size == "128,128"


def test_vae_image_size_missing_comma():
    with pytest.raises(ValidationError):
        VAEConfig(image_size="128")


def test_vae_image_size_empty():
    with pytest.raises(ValidationError):
        VAEConfig(image_size="")


# ── UNetConfig validators ────────────────────────────────────────────────────

def test_unet_block_out_channels_no_comma(valid_config_dict):
    valid_config_dict["unet"]["block_out_channels"] = "64"
    with pytest.raises(ValidationError):
        PipelineConfig(**valid_config_dict)


def test_unet_negative_in_channels(valid_config_dict):
    valid_config_dict["unet"]["in_channels"] = -1
    with pytest.raises(ValidationError):
        PipelineConfig(**valid_config_dict)


def test_unet_dropout_out_of_range(valid_config_dict):
    valid_config_dict["unet"]["dropout"] = 1.5
    with pytest.raises(ValidationError):
        PipelineConfig(**valid_config_dict)


# ── TrainingConfig ───────────────────────────────────────────────────────────

def test_training_cfg_dropout_out_of_range(valid_config_dict):
    valid_config_dict["training"]["cfg_dropout_prob"] = -0.1
    with pytest.raises(ValidationError):
        PipelineConfig(**valid_config_dict)


def test_training_batch_size_zero(valid_config_dict):
    valid_config_dict["training"]["batch_size"] = 0
    with pytest.raises(ValidationError):
        PipelineConfig(**valid_config_dict)


# ── DDPMSchedulerConfig ──────────────────────────────────────────────────────

def test_ddpm_scheduler_defaults():
    cfg = DDPMSchedulerConfig(T=1000)
    assert cfg.beta_schedule == "squaredcos_cap_v2"


def test_ddpm_scheduler_zero_T():
    with pytest.raises(ValidationError):
        DDPMSchedulerConfig(T=0)


# ── TaskResult ───────────────────────────────────────────────────────────────

def test_task_result_without_artifacts():
    result = TaskResult(
        task_name="upload",
        status="success",
        message="done",
        execution_time=1.5,
    )
    assert result.artifacts is None


def test_task_result_with_artifacts():
    result = TaskResult(
        task_name="upload",
        status="success",
        message="done",
        artifacts={"s3_path": "s3://bucket/key"},
        execution_time=1.5,
    )
    assert result.artifacts["s3_path"] == "s3://bucket/key"


def test_task_result_dict_serialisation():
    result = TaskResult(
        task_name="upload",
        status="failed",
        message="error",
        execution_time=0.1,
    )
    d = result.dict()
    assert d["status"] == "failed"
    assert "artifacts" in d

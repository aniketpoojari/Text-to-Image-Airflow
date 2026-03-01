"""
Tests for ModelConverter.
torch.onnx.export and all model downloads are mocked — no GPU required.
"""
import os
import sys
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def converter():
    """ModelConverter instantiated with device forced to cpu."""
    with patch("model_converter.torch.cuda.is_available", return_value=False):
        from model_converter import ModelConverter
        return ModelConverter()


@pytest.fixture
def fake_onnx_files(tmp_path):
    """
    Pre-create the expected .onnx files inside tmp_path/onnx_models/ so
    convert_to_torchserve believes they were exported successfully.
    """
    onnx_dir = tmp_path / "onnx_models"
    onnx_dir.mkdir()
    for name in ["clip_text_encoder.onnx", "vae_decoder.onnx", "unet.onnx"]:
        (onnx_dir / name).write_bytes(b"fake")
    return tmp_path


# ── Directory creation ────────────────────────────────────────────────────────

def test_output_directories_are_created(converter, tmp_path):
    """convert_to_torchserve must create output_dir and onnx_models/ subdir."""
    with patch.object(converter, "_export_clip_text_encoder"), \
         patch.object(converter, "_export_vae_decoder"), \
         patch.object(converter, "_export_unet"):
        converter.convert_to_torchserve(
            diffuser_path="dummy.pth",
            output_dir=str(tmp_path / "out"),
            image_size=(128, 128),
        )
    assert os.path.isdir(tmp_path / "out")
    assert os.path.isdir(tmp_path / "out" / "onnx_models")


# ── Return value structure ────────────────────────────────────────────────────

def test_convert_returns_expected_keys(converter, fake_onnx_files):
    """Return dict must contain the documented keys."""
    with patch.object(converter, "_export_clip_text_encoder"), \
         patch.object(converter, "_export_vae_decoder"), \
         patch.object(converter, "_export_unet"):
        result = converter.convert_to_torchserve(
            diffuser_path="dummy.pth",
            output_dir=str(fake_onnx_files),
            image_size=(128, 128),
        )
    assert "conversion_status" in result
    assert "output_dir" in result
    assert "created_models" in result
    assert "total_models" in result
    assert "model_paths" in result


def test_convert_reports_all_three_models(converter, fake_onnx_files):
    """When all three .onnx files exist, created_models should have 3 entries."""
    with patch.object(converter, "_export_clip_text_encoder"), \
         patch.object(converter, "_export_vae_decoder"), \
         patch.object(converter, "_export_unet"):
        result = converter.convert_to_torchserve(
            diffuser_path="dummy.pth",
            output_dir=str(fake_onnx_files),
            image_size=(128, 128),
        )
    assert result["total_models"] == 3
    assert set(result["created_models"]) == {
        "clip_text_encoder.onnx", "vae_decoder.onnx", "unet.onnx"
    }


def test_convert_handles_missing_onnx_files(converter, tmp_path):
    """If export fails silently, missing files are not included in created_models."""
    with patch.object(converter, "_export_clip_text_encoder"), \
         patch.object(converter, "_export_vae_decoder"), \
         patch.object(converter, "_export_unet"):
        result = converter.convert_to_torchserve(
            diffuser_path="dummy.pth",
            output_dir=str(tmp_path),
            image_size=(128, 128),
        )
    assert result["total_models"] == 0
    assert result["created_models"] == []


# ── Latent size derivation ────────────────────────────────────────────────────

def test_latent_size_derived_from_image_size(converter, tmp_path):
    """_export_vae_decoder must be called with the correct latent_h/latent_w."""
    with patch.object(converter, "_export_clip_text_encoder"), \
         patch.object(converter, "_export_unet"):
        mock_vae_export = MagicMock()
        converter._export_vae_decoder = mock_vae_export
        converter.convert_to_torchserve(
            diffuser_path="dummy.pth",
            output_dir=str(tmp_path),
            image_size=(128, 128),   # 128 // 8 = 16
        )
    mock_vae_export.assert_called_once_with(
        os.path.join(str(tmp_path), "onnx_models"), 16, 16
    )

"""
Tests for all five orchestrators.
All underlying utils (DataUploader, TrainingManager, etc.) are mocked.
No AWS, SageMaker, or GPU required.
"""
import sys
import yaml
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "orchestrators"))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

CONFIG_PATH = str(Path(__file__).parent.parent / "config" / "pipeline_config.yaml")


# sagemaker is pre-mocked in conftest.py via sys.modules before any imports


# ── DataOrchestrator ──────────────────────────────────────────────────────────

class TestDataOrchestrator:
    def _make(self, upload_return=None, upload_raises=None):
        mock_uploader = MagicMock()
        if upload_raises:
            mock_uploader.upload_data.side_effect = upload_raises
        else:
            mock_uploader.upload_data.return_value = upload_return or {
                "upload_status": "completed",
                "s3_path": "s3://bucket/key",
                "local_path": "data/raw/flowers.zip",
                "file_size": 1024,
            }
        with patch("orchestrators.data_orchestrator.DataUploader", return_value=mock_uploader):
            from orchestrators.data_orchestrator import DataOrchestrator
            orch = DataOrchestrator(CONFIG_PATH)
            orch.data_uploader = mock_uploader
        return orch, mock_uploader

    def test_success_returns_success_status(self):
        orch, _ = self._make()
        result = orch.execute_upload()
        assert result["status"] == "success"
        assert result["task_name"] == "data_upload"

    def test_success_has_artifacts(self):
        orch, _ = self._make()
        result = orch.execute_upload()
        assert result["artifacts"] is not None

    def test_failure_returns_failed_status(self):
        orch, _ = self._make(upload_raises=Exception("S3 error"))
        result = orch.execute_upload()
        assert result["status"] == "failed"
        assert "S3 error" in result["message"]

    def test_uploader_called_with_correct_paths(self):
        orch, mock_uploader = self._make()
        orch.execute_upload()
        mock_uploader.upload_data.assert_called_once()
        call_args = mock_uploader.upload_data.call_args
        assert "s3://" in call_args[1]["s3_path"] or "s3://" in str(call_args)

    def test_execution_time_is_recorded(self):
        orch, _ = self._make()
        result = orch.execute_upload()
        assert result["execution_time"] is not None
        assert result["execution_time"] >= 0


# ── TrainingOrchestrator ──────────────────────────────────────────────────────

class TestTrainingOrchestrator:
    def _make(self, train_return=None, train_raises=None):
        mock_manager = MagicMock()
        if train_raises:
            mock_manager.run_training_job.side_effect = train_raises
        else:
            mock_manager.run_training_job.return_value = train_return or {
                "training_status": "completed",
                "job_name": "test-job-001",
                "model_artifacts": "s3://bucket/model.tar.gz",
                "instance_type": "ml.g4dn.xlarge",
                "instance_count": 1,
            }
        with patch("orchestrators.training_orchestrator.TrainingManager", return_value=mock_manager):
            from orchestrators.training_orchestrator import TrainingOrchestrator
            orch = TrainingOrchestrator(CONFIG_PATH)
            orch.training_manager = mock_manager
        return orch, mock_manager

    def test_success_returns_success_status(self):
        orch, _ = self._make()
        result = orch.execute_training()
        assert result["status"] == "success"
        assert result["task_name"] == "training_job"

    def test_failure_returns_failed_status(self):
        orch, _ = self._make(train_raises=Exception("SageMaker timeout"))
        result = orch.execute_training()
        assert result["status"] == "failed"
        assert "SageMaker timeout" in result["message"]

    def test_manager_called_with_config(self):
        orch, mock_manager = self._make()
        orch.execute_training()
        mock_manager.run_training_job.assert_called_once_with(orch.config)


# ── ModelOrchestrator ─────────────────────────────────────────────────────────

class TestModelOrchestrator:
    def _make(self, download_return=None, download_raises=None):
        mock_downloader = MagicMock()
        if download_raises:
            mock_downloader.download_best_model.side_effect = download_raises
        else:
            mock_downloader.download_best_model.return_value = download_return or {
                "download_status": "completed",
                "best_run_id": "abc123",
                "diffuser_path": "models/diffuser.pth",
                "diffuser_loss": 0.42,
                "diffuser_size_mb": 50.0,
            }
        with patch("utils.model_downloader.boto3"), \
             patch("orchestrators.model_orchestrator.ModelDownloader", return_value=mock_downloader):
            from orchestrators.model_orchestrator import ModelOrchestrator
            orch = ModelOrchestrator(CONFIG_PATH)
            orch.model_downloader = mock_downloader
        return orch, mock_downloader

    def test_success_returns_success_status(self):
        orch, _ = self._make()
        result = orch.execute_download()
        assert result["status"] == "success"
        assert result["task_name"] == "model_download"

    def test_artifacts_contain_download_result(self):
        orch, _ = self._make()
        result = orch.execute_download()
        assert result["artifacts"]["best_run_id"] == "abc123"

    def test_failure_on_missing_experiment(self):
        orch, _ = self._make(download_raises=Exception("Experiment not found"))
        result = orch.execute_download()
        assert result["status"] == "failed"


# ── ConversionOrchestrator ───────────────────────────────────────────────────

class TestConversionOrchestrator:
    def _make(self, convert_return=None, convert_raises=None):
        mock_converter = MagicMock()
        if convert_raises:
            mock_converter.convert_to_torchserve.side_effect = convert_raises
        else:
            mock_converter.convert_to_torchserve.return_value = convert_return or {
                "conversion_status": "completed",
                "output_dir": "models/onnx_models",
                "created_models": ["clip_text_encoder.onnx", "vae_decoder.onnx", "unet.onnx"],
                "total_models": 3,
                "model_paths": [],
            }
        with patch("orchestrators.conversion_orchestrator.ModelConverter", return_value=mock_converter):
            from orchestrators.conversion_orchestrator import ConversionOrchestrator
            orch = ConversionOrchestrator(CONFIG_PATH)
            orch.model_converter = mock_converter
        return orch, mock_converter

    def test_success_returns_success_status(self):
        orch, _ = self._make()
        result = orch.execute_conversion()
        assert result["status"] == "success"
        assert result["task_name"] == "model_conversion"

    def test_image_size_passed_correctly(self):
        orch, mock_converter = self._make()
        orch.execute_conversion()
        call_kwargs = mock_converter.convert_to_torchserve.call_args[1]
        assert call_kwargs["image_size"] == (128, 128)

    def test_failure_is_caught(self):
        orch, _ = self._make(convert_raises=RuntimeError("CUDA OOM"))
        result = orch.execute_conversion()
        assert result["status"] == "failed"
        assert "CUDA OOM" in result["message"]


# ── CaptionOrchestrator ───────────────────────────────────────────────────────

class TestCaptionOrchestrator:
    def _make(self, gen_return=None, gen_raises=None):
        mock_generator = MagicMock()
        if gen_raises:
            mock_generator.generate_captions.side_effect = gen_raises
        else:
            mock_generator.generate_captions.return_value = gen_return or {
                "generated": 10,
                "skipped": 5,
                "total": 15,
            }
        with patch("orchestrators.caption_orchestrator.CaptionGenerator", return_value=mock_generator):
            from orchestrators.caption_orchestrator import CaptionOrchestrator
            orch = CaptionOrchestrator(CONFIG_PATH)
            orch.caption_generator = mock_generator
        return orch, mock_generator

    def test_success_returns_success_status(self):
        orch, _ = self._make()
        result = orch.execute_caption_generation()
        assert result["status"] == "success"
        assert result["task_name"] == "caption_generation"

    def test_generated_count_in_message(self):
        orch, _ = self._make()
        result = orch.execute_caption_generation()
        assert "10" in result["message"]  # generated count

    def test_generator_called_with_config(self):
        orch, mock_gen = self._make()
        orch.execute_caption_generation()
        mock_gen.generate_captions.assert_called_once_with(orch.config)

    def test_failure_is_caught(self):
        orch, _ = self._make(gen_raises=Exception("Florence OOM"))
        result = orch.execute_caption_generation()
        assert result["status"] == "failed"
        assert "Florence OOM" in result["message"]

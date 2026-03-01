import mlflow
import pandas as pd
import boto3
import os
from typing import Dict, Any
from orchestrators.config_models import PipelineConfig
from botocore.exceptions import ClientError


class ModelDownloader:
    """Core functionality for model download operations.

    The VAE is frozen during training (loaded from stabilityai/sd-vae-ft-mse)
    and is never saved to S3, so only the diffuser (UNet) checkpoint is
    downloaded here.
    """

    def __init__(self):
        self.s3_client = boto3.client('s3')

    def download_best_model(self, config: PipelineConfig) -> Dict[str, Any]:
        """Download the best diffuser checkpoint from the MLflow S3 bucket."""

        mlflow.set_tracking_uri(config.mlflow.server_uri)

        experiment = mlflow.get_experiment_by_name(config.mlflow.experiment_name)
        if experiment is None:
            raise Exception(f"Experiment '{config.mlflow.experiment_name}' not found")

        runs_df = pd.DataFrame(mlflow.search_runs(experiment_ids=[experiment.experiment_id]))

        if runs_df.empty:
            raise Exception(f"No runs found in experiment '{config.mlflow.experiment_name}'")

        finished_runs = runs_df[runs_df["status"] == "FINISHED"]
        if finished_runs.empty:
            raise Exception("No finished runs found in the experiment")

        if "metrics.val_diffuser_loss" not in finished_runs.columns:
            raise Exception("No val_diffuser_loss metric found in runs")

        best_run = finished_runs.loc[finished_runs["metrics.val_diffuser_loss"].idxmin()]
        best_run_id = best_run['run_id']

        print(f"Best run ID: {best_run_id}")
        print(f"Best val_diffuser_loss: {best_run['metrics.val_diffuser_loss']:.4f}")

        bucket_name = config.mlflow.s3_mlruns_bucket
        diffuser_s3_key = f"{best_run_id}/diffuser.pth"
        diffuser_local_path = "models/diffuser.pth"

        os.makedirs("models", exist_ok=True)

        print(f"Downloading diffuser from s3://{bucket_name}/{diffuser_s3_key}")
        self.s3_client.download_file(bucket_name, diffuser_s3_key, diffuser_local_path)

        if not os.path.exists(diffuser_local_path):
            raise Exception(f"Download failed: {diffuser_local_path} not found")

        return {
            "download_status": "completed",
            "best_run_id": best_run_id,
            "diffuser_path": diffuser_local_path,
            "diffuser_loss": float(best_run['metrics.val_diffuser_loss']),
            "diffuser_size_mb": round(os.path.getsize(diffuser_local_path) / (1024 * 1024), 2),
        }

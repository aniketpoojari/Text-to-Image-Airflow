import mlflow
import pandas as pd
import boto3
import os
from typing import Dict, Any
from orchestrators.config_models import PipelineConfig
from botocore.exceptions import ClientError

class ModelDownloader:
    """Core functionality for model download operations"""
    
    def __init__(self):
        self.s3_client = boto3.client('s3')
    
    def download_best_model(self, config: PipelineConfig) -> Dict[str, Any]:
        """Download the best model from MLflow tracking server"""
        
        try:
            # Set up MLflow tracking
            mlflow.set_tracking_uri(config.mlflow.server_uri)
            
            # Get experiment
            experiment = mlflow.get_experiment_by_name(config.mlflow.experiment_name)
            if experiment is None:
                raise Exception(f"Experiment '{config.mlflow.experiment_name}' not found")
            
            experiment_id = experiment.experiment_id
            
            # Search for runs in the experiment
            runs_df = pd.DataFrame(mlflow.search_runs(experiment_ids=[experiment_id]))
            
            if runs_df.empty:
                raise Exception(f"No runs found in experiment '{config.mlflow.experiment_name}'")
            
            # Filter for finished runs
            finished_runs = runs_df[runs_df["status"] == "FINISHED"]
            
            if finished_runs.empty:
                raise Exception("No finished runs found in the experiment")
            
            # Find the best run based on validation VAE loss
            if "metrics.val_vae_loss" not in finished_runs.columns:
                raise Exception("No validation VAE loss metric found in runs")
            
            best_run = finished_runs.loc[finished_runs["metrics.val_vae_loss"].idxmin()]
            best_run_id = best_run['run_id']
            
            print(f"Best run ID: {best_run_id}")
            print(f"Best VAE loss: {best_run['metrics.val_vae_loss']}")
            
            # Download models from S3
            bucket_name = config.mlflow.s3_mlruns_bucket
            
            # Define S3 keys for the models
            vae_s3_key = f"{best_run_id}/vae.pth"
            diffuser_s3_key = f"{best_run_id}/diffuser.pth"
            
            # Define local paths
            vae_local_path = "saved_models/vae.pth"
            diffuser_local_path = "saved_models/diffuser.pth"
            
            # Create saved_models directory if it doesn't exist
            os.makedirs("saved_models", exist_ok=True)
            
            # Download VAE model
            print(f"Downloading VAE model from s3://{bucket_name}/{vae_s3_key}")
            self.s3_client.download_file(bucket_name, vae_s3_key, vae_local_path)
            
            # Download Diffuser model
            print(f"Downloading Diffuser model from s3://{bucket_name}/{diffuser_s3_key}")
            self.s3_client.download_file(bucket_name, diffuser_s3_key, diffuser_local_path)
            
            # Verify downloads
            if not os.path.exists(vae_local_path):
                raise Exception(f"VAE model download failed: {vae_local_path} not found")
            
            if not os.path.exists(diffuser_local_path):
                raise Exception(f"Diffuser model download failed: {diffuser_local_path} not found")
            
            return {
                "download_status": "completed",
                "best_run_id": best_run_id,
                "vae_path": vae_local_path,
                "diffuser_path": diffuser_local_path,
                "vae_loss": float(best_run['metrics.val_vae_loss']),
                "diffuser_loss": float(best_run.get('metrics.val_diffuser_loss', 0)),
                "vae_size_mb": round(os.path.getsize(vae_local_path) / (1024 * 1024), 2),
                "diffuser_size_mb": round(os.path.getsize(diffuser_local_path) / (1024 * 1024), 2)
            }
            
        except ClientError as e:
            raise Exception(f"AWS S3 download failed: {e}")
        except Exception as e:
            raise Exception(f"Model download failed: {e}")

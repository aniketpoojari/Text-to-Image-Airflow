import sagemaker
from sagemaker.pytorch import PyTorch
from datetime import datetime
import tempfile
import yaml
import os
from typing import Dict, Any
from orchestrators.config_models import PipelineConfig

class TrainingManager:
    """Core functionality for training job management"""
    
    def __init__(self):
        self.sagemaker_session = sagemaker.Session()
    
    def run_training_job(self, config: PipelineConfig) -> Dict[str, Any]:
        """Run SageMaker training job with the provided configuration"""
        
        try:
            # Create environment variables for the training container
            environment = {
                "TRAIN_SIZE": str(config.data.train_size),
                "VAL_SIZE": str(config.data.val_size),
                "VAE_IMAGE_SIZE": config.vae.image_size,
                "MAX_LENGTH": str(config.data.max_length),
                "BATCH_SIZE": str(config.data.batch_size),
                "T": str(config.ddpm_scheduler.T),
                "UNET_IMAGE_SIZE": config.unet.image_size,
                "IN_CHANNELS": str(config.unet.in_channels),
                "OUT_CHANNELS": str(config.unet.out_channels),
                "DOWN_BLOCK_TYPES": config.unet.down_block_types,
                "UP_BLOCK_TYPES": config.unet.up_block_types,
                "MID_BLOCK_TYPE": config.unet.mid_block_type,
                "BLOCK_OUT_CHANNELS": config.unet.block_out_channels,
                "LAYERS_PER_BLOCK": str(config.unet.layers_per_block),
                "NORM_NUM_GROUPS": str(config.unet.norm_num_groups),
                "CROSS_ATTENTION_DIM": str(config.unet.cross_attention_dim),
                "ATTENTION_HEAD_DIM": str(config.unet.attention_head_dim),
                "DROPOUT": str(config.unet.dropout),
                "TIME_EMBEDDING_TYPE": config.unet.time_embedding_type,
                "ACT_FN": config.unet.act_fn,
                "VAE_LEARNING_RATE": str(config.vae.learning_rate),
                "UNET_LEARNING_RATE": str(config.unet.learning_rate),
                "WEIGHT_DECAY": str(config.training.weight_decay),
                "NUM_EPOCHS": str(config.training.num_epochs),
                "EXPERIMENT_NAME": config.mlflow.experiment_name,
                "RUN_NAME": config.mlflow.run_name,
                "REGISTERED_MODEL_NAME": config.mlflow.registered_model_name,
                "SERVER_URI": config.mlflow.server_uri,
                "S3_MLRUNS_BUCKET": config.mlflow.s3_mlruns_bucket,
                "MLFLOW_TRACKING_USERNAME": config.mlflow.tracking_username,
                "MLFLOW_TRACKING_PASSWORD": config.mlflow.tracking_password,
            }
            
            # Configure PyTorch estimator
            estimator = PyTorch(
                entry_point=config.sagemaker.entry_point,
                source_dir=config.sagemaker.source_dir,
                role=config.sagemaker.role,
                framework_version=config.sagemaker.framework_version,
                py_version=config.sagemaker.py_version,
                instance_count=config.sagemaker.instance_count,
                instance_type=config.sagemaker.instance_type,
                use_spot_instances=config.sagemaker.use_spot_instances,
                max_wait=config.sagemaker.max_wait,
                max_run=config.sagemaker.max_run,
                environment=environment,
                distribution={
                    "deepspeed": {
                        "enabled": True
                    }
                },
                sagemaker_session=self.sagemaker_session
            )
            
            # Define data channels
            data_channels = {
                'train': config.sagemaker.s3_train_data,
            }
            
            print("Starting SageMaker training job...")
            print(f"Instance type: {config.sagemaker.instance_type}")
            print(f"Instance count: {config.sagemaker.instance_count}")
            print(f"Training data: {config.sagemaker.s3_train_data}")
            
            # Start training job
            estimator.fit(inputs=data_channels, logs="minimal")
            
            # Create training completion marker
            self._create_completion_marker()
            
            return {
                "training_status": "completed",
                "job_name": estimator.latest_training_job.name,
                "model_artifacts": estimator.model_data,
                "training_image": estimator.training_image_uri(),
                "instance_type": config.sagemaker.instance_type,
                "instance_count": config.sagemaker.instance_count
            }
            
        except Exception as e:
            raise Exception(f"SageMaker training job failed: {e}")
    
    def _create_completion_marker(self):
        """Create a completion marker file"""
        try:
            with open("training_completion.txt", "w") as file:
                current_datetime = datetime.now()
                formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
                file.write(f"Training Completed at {formatted_datetime}")
        except Exception as e:
            print(f"Warning: Could not create completion marker: {e}")

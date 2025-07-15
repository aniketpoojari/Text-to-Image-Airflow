from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
from datetime import datetime

class DataConfig(BaseModel):
    """Configuration for data processing"""
    train_size: int = Field(gt=0, description="Number of training samples")
    val_size: int = Field(gt=0, description="Number of validation samples")
    raw_data_path: str = Field(description="Path to raw data")

class VAEConfig(BaseModel):
    """Configuration for VAE model"""
    image_size: str = Field(description="VAE image size as comma-separated string")
    learning_rate: float = Field(gt=0, description="VAE learning rate")
    
    @validator('image_size')
    def validate_image_size(cls, v):
        
        if not v or ',' not in v:
            raise ValueError('Must be comma-separated string')
        return v

class UNetConfig(BaseModel):
    """Configuration for UNet model"""
    image_size: str = Field(description="UNet image size as comma-separated string")
    in_channels: int = Field(gt=0, description="Input channels")
    out_channels: int = Field(gt=0, description="Output channels")
    down_block_types: str = Field(description="Down block types as comma-separated string")
    up_block_types: str = Field(description="Up block types as comma-separated string")
    mid_block_type: str = Field(description="Middle block type")
    block_out_channels: str = Field(description="Block output channels as comma-separated string")
    layers_per_block: int = Field(gt=0, description="Layers per block")
    norm_num_groups: int = Field(gt=0, description="Normalization groups")
    cross_attention_dim: int = Field(gt=0, description="Cross attention dimension")
    attention_head_dim: int = Field(gt=0, description="Attention head dimension")
    dropout: float = Field(ge=0.0, le=1.0, description="Dropout rate")
    time_embedding_type: str = Field(description="Time embedding type")
    act_fn: str = Field(description="Activation function")
    learning_rate: float = Field(gt=0, description="UNet learning rate")
    
    @validator('image_size', 'down_block_types', 'up_block_types', 'block_out_channels')
    def validate_comma_separated(cls, v):
        if not v or ',' not in v:
            raise ValueError('Must be comma-separated string')
        return v

class DDPMSchedulerConfig(BaseModel):
    """Configuration for DDPM Scheduler"""
    T: int = Field(gt=0, description="Number of diffusion timesteps")

class CLIPConfig(BaseModel):
    """Configuration for CLIP model"""
    max_length: int = Field(gt=0, description="Maximum text length")

class TrainingConfig(BaseModel):
    """Configuration for training parameters"""
    batch_size: int = Field(gt=0, description="Training batch size")
    weight_decay: float = Field(ge=0, description="Weight decay")
    num_epochs: int = Field(gt=0, description="Number of training epochs")

class MLflowConfig(BaseModel):
    """Configuration for MLflow tracking"""
    experiment_name: str = Field(description="MLflow experiment name")
    run_name: str = Field(description="MLflow run name")
    registered_model_name: str = Field(description="Registered model name")
    server_uri: str = Field(description="MLflow server URI")
    s3_mlruns_bucket: str = Field(description="S3 bucket for MLflow artifacts")
    tracking_username: str = Field(description="MLflow tracking username")
    tracking_password: str = Field(description="MLflow tracking password")

class SageMakerConfig(BaseModel):
    """Configuration for SageMaker training"""
    role: str = Field(description="SageMaker execution role")
    instance_count: int = Field(gt=0, description="Number of training instances")
    instance_type: str = Field(description="Training instance type")
    framework_version: str = Field(description="PyTorch framework version")
    py_version: str = Field(description="Python version")
    max_wait: int = Field(gt=0, description="Maximum wait time")
    max_run: int = Field(gt=0, description="Maximum run time")
    use_spot_instances: bool = Field(description="Use spot instances")
    s3_train_data: str = Field(description="S3 path to training data")
    entry_point: str = Field(description="Training script entry point")
    source_dir: str = Field(description="Source directory")

class PipelineConfig(BaseModel):
    """Main configuration combining all components"""
    data: DataConfig
    vae: VAEConfig
    unet: UNetConfig
    ddpm_scheduler: DDPMSchedulerConfig
    clip: CLIPConfig
    training: TrainingConfig
    mlflow: MLflowConfig
    sagemaker: SageMakerConfig

class TaskResult(BaseModel):
    """Result model for task execution"""
    task_name: str = Field(description="Task name")
    status: str = Field(description="Task status")
    message: str = Field(description="Task message")
    execution_time: Optional[float] = Field(description="Execution time in seconds")

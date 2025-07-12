import yaml
import time
from .config_models import PipelineConfig, TaskResult
from utils.data_uploader import DataUploader

class DataOrchestrator:
    """Orchestrator for data upload operations"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_and_validate_config()
        self.data_uploader = DataUploader()
    
    def _load_and_validate_config(self) -> PipelineConfig:
        """Load and validate configuration using Pydantic"""
        with open(self.config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        
        # Validate with Pydantic
        validated_config = PipelineConfig(**config_data)
        return validated_config
    
    def execute_upload(self) -> dict:
        """Execute data upload with config validation"""
        start_time = time.time()
        
        try:
            # Call utils layer with validated config
            result = self.data_uploader.upload_data(
                local_path=self.config.data.raw_data_path,
                s3_path=self.config.sagemaker.s3_train_data
            )
            
            execution_time = time.time() - start_time
            
            task_result = TaskResult(
                task_name="data_upload",
                status="success",
                message="Data uploaded successfully",
                artifacts={"s3_path": self.config.sagemaker.s3_train_data},
                execution_time=execution_time
            )
            
            return task_result.dict()
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            task_result = TaskResult(
                task_name="data_upload",
                status="failed",
                message=f"Data upload failed: {str(e)}",
                execution_time=execution_time
            )
            
            return task_result.dict()

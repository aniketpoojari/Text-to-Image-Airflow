import yaml
import time
from .config_models import PipelineConfig, TaskResult
from utils.model_downloader import ModelDownloader

class ModelOrchestrator:
    """Orchestrator for model download operations"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_and_validate_config()
        self.model_downloader = ModelDownloader()
    
    def _load_and_validate_config(self) -> PipelineConfig:
        """Load and validate configuration using Pydantic"""
        with open(self.config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        
        # Validate with Pydantic
        validated_config = PipelineConfig(**config_data)
        return validated_config
    
    def execute_download(self) -> dict:
        """Execute model download with config validation"""
        start_time = time.time()
        
        try:
            # Call utils layer with validated config
            result = self.model_downloader.download_best_model(self.config)
            
            execution_time = time.time() - start_time
            
            task_result = TaskResult(
                task_name="model_download",
                status="success",
                message="Model downloaded successfully",
                artifacts=result,
                execution_time=execution_time
            )
            
            return task_result.dict()
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            task_result = TaskResult(
                task_name="model_download",
                status="failed",
                message=f"Model download failed: {str(e)}",
                execution_time=execution_time
            )
            
            return task_result.dict()

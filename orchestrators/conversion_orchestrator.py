import yaml
import time
from .config_models import PipelineConfig, TaskResult
from utils.model_converter import ModelConverter

class ConversionOrchestrator:
    """Orchestrator for model conversion operations"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_and_validate_config()
        self.model_converter = ModelConverter()
    
    def _load_and_validate_config(self) -> PipelineConfig:
        """Load and validate configuration using Pydantic"""
        with open(self.config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        
        # Validate with Pydantic
        validated_config = PipelineConfig(**config_data)
        return validated_config
    
    def execute_conversion(self) -> dict:
        """Execute model conversion with config validation"""
        start_time = time.time()
        
        try:
            # Call utils layer with validated config
            result = self.model_converter.convert_to_torchserve(
                vae_path="saved_models/vae.pth",
                diffuser_path="saved_models/diffuser.pth",
                output_dir="saved_models/torchserve"
            )
            
            execution_time = time.time() - start_time
            
            task_result = TaskResult(
                task_name="model_conversion",
                status="success",
                message="Model conversion completed successfully",
                artifacts=result,
                execution_time=execution_time
            )
            
            return task_result.dict()
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            task_result = TaskResult(
                task_name="model_conversion",
                status="failed",
                message=f"Model conversion failed: {str(e)}",
                execution_time=execution_time
            )
            
            return task_result.dict()

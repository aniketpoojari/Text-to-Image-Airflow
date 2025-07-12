import yaml
import time
from .config_models import PipelineConfig, TaskResult
from utils.training_manager import TrainingManager

class TrainingOrchestrator:
    """Orchestrator for training operations"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_and_validate_config()
        self.training_manager = TrainingManager()
    
    def _load_and_validate_config(self) -> PipelineConfig:
        """Load and validate configuration using Pydantic"""
        with open(self.config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        
        # Validate with Pydantic
        validated_config = PipelineConfig(**config_data)
        return validated_config
    
    def execute_training(self) -> dict:
        """Execute training job with config validation"""
        start_time = time.time()
        
        try:
            # Call utils layer with validated config
            result = self.training_manager.run_training_job(self.config)
            
            execution_time = time.time() - start_time
            
            task_result = TaskResult(
                task_name="training_job",
                status="success",
                message="Training job completed successfully",
                artifacts={"training_result": result},
                execution_time=execution_time
            )
            
            return task_result.dict()
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            task_result = TaskResult(
                task_name="training_job",
                status="failed",
                message=f"Training job failed: {str(e)}",
                execution_time=execution_time
            )
            
            return task_result.dict()

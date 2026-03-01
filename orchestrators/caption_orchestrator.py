import yaml
import time
from .config_models import PipelineConfig, TaskResult
from utils.caption_generator import CaptionGenerator


class CaptionOrchestrator:
    """Orchestrator for Florence-2 caption generation"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_and_validate_config()
        self.caption_generator = CaptionGenerator()

    def _load_and_validate_config(self) -> PipelineConfig:
        with open(self.config_path, 'r') as file:
            config_data = yaml.safe_load(file)
        return PipelineConfig(**config_data)

    def execute_caption_generation(self) -> dict:
        start_time = time.time()

        try:
            result = self.caption_generator.generate_captions(self.config)

            task_result = TaskResult(
                task_name="caption_generation",
                status="success",
                message=f"Caption generation complete. Generated {result['generated']} captions.",
                artifacts=result,
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            task_result = TaskResult(
                task_name="caption_generation",
                status="failed",
                message=f"Caption generation failed: {str(e)}",
                execution_time=time.time() - start_time,
            )

        return task_result.dict()

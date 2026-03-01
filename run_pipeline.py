"""
Run the full ML pipeline without Airflow.
Usage: python run_pipeline.py
"""
from orchestrators.caption_orchestrator import CaptionOrchestrator
from orchestrators.data_orchestrator import DataOrchestrator
from orchestrators.training_orchestrator import TrainingOrchestrator
from orchestrators.model_orchestrator import ModelOrchestrator
from orchestrators.conversion_orchestrator import ConversionOrchestrator

CONFIG_PATH = "config/pipeline_config.yaml"

steps = [
    ("Generate Captions",  CaptionOrchestrator,  "execute_caption_generation"),
    ("Upload Data",        DataOrchestrator,      "execute_upload"),
    ("Train on SageMaker", TrainingOrchestrator,  "execute_training"),
    ("Download Model",     ModelOrchestrator,     "execute_download"),
    ("Convert to ONNX",    ConversionOrchestrator,"execute_conversion"),
]

for name, OrchestratorClass, method in steps:
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    result = getattr(OrchestratorClass(CONFIG_PATH), method)()
    print(result)
    if result["status"] == "failed":
        print(f"\nPipeline stopped at: {name}")
        break

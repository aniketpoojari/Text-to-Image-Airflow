from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

from orchestrators.caption_orchestrator import CaptionOrchestrator
from orchestrators.data_orchestrator import DataOrchestrator
from orchestrators.training_orchestrator import TrainingOrchestrator
from orchestrators.model_orchestrator import ModelOrchestrator
from orchestrators.conversion_orchestrator import ConversionOrchestrator

CONFIG_PATH = "config/pipeline_config.yaml"

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='End-to-end diffusion model training pipeline',
    schedule_interval='@weekly',
    catchup=False,
    tags=['ml', 'sagemaker', 'diffusion', 'airflow'],
)

# Task 0: Generate captions with Florence-2 (skips if captions already exist)
def caption_generation_task():
    orchestrator = CaptionOrchestrator(CONFIG_PATH)
    return orchestrator.execute_caption_generation()

generate_captions = PythonOperator(
    task_id='generate_captions',
    python_callable=caption_generation_task,
    dag=dag,
)

# Task 1: Upload data to S3
def upload_data_task():
    orchestrator = DataOrchestrator(CONFIG_PATH)
    return orchestrator.execute_upload()

upload_data = PythonOperator(
    task_id='upload_data_to_s3',
    python_callable=upload_data_task,
    dag=dag,
)

# Task 2: Run SageMaker training job
def run_training_task():
    orchestrator = TrainingOrchestrator(CONFIG_PATH)
    return orchestrator.execute_training()

training_job = PythonOperator(
    task_id='sagemaker_training_job',
    python_callable=run_training_task,
    dag=dag,
)

# Task 3: Download best model from S3
def download_model_task():
    orchestrator = ModelOrchestrator(CONFIG_PATH)
    return orchestrator.execute_download()

download_model = PythonOperator(
    task_id='download_best_model',
    python_callable=download_model_task,
    dag=dag,
)

# Task 4: Convert to ONNX for TorchServe
def convert_model_task():
    orchestrator = ConversionOrchestrator(CONFIG_PATH)
    return orchestrator.execute_conversion()

convert_model = PythonOperator(
    task_id='convert_to_torchserve',
    python_callable=convert_model_task,
    dag=dag,
)

# Pipeline dependency chain
generate_captions >> upload_data >> training_job >> download_model >> convert_model

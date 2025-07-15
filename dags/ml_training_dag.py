from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

from orchestrators.data_orchestrator import DataOrchestrator
from orchestrators.training_orchestrator import TrainingOrchestrator
from orchestrators.model_orchestrator import ModelOrchestrator
from orchestrators.conversion_orchestrator import ConversionOrchestrator

# Configuration file path
CONFIG_PATH = "config/pipeline_config.yaml"

# Default arguments
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='ML Training Pipeline with Three-Layer Architecture',
    schedule_interval='@weekly',
    catchup=False,
    tags=['ml', 'sagemaker', 'diffusion', 'three-layer']
)

# Task 1: Upload data
def upload_data_task():
    orchestrator = DataOrchestrator(CONFIG_PATH)
    return orchestrator.execute_upload()

upload_data = PythonOperator(
    task_id='upload_data_to_s3',
    python_callable=upload_data_task,
    dag=dag
)

# Task 2: Run training job
def run_training_task():
    orchestrator = TrainingOrchestrator(CONFIG_PATH)
    return orchestrator.execute_training()

training_job = PythonOperator(
    task_id='sagemaker_training_job',
    python_callable=run_training_task,
    dag=dag
)

# Task 3: Download best model
def download_model_task():
    orchestrator = ModelOrchestrator(CONFIG_PATH)
    return orchestrator.execute_download()

download_model = PythonOperator(
    task_id='download_best_model',
    python_callable=download_model_task,
    dag=dag
)

# Task 4: Convert to TorchServe
def convert_model_task():
    orchestrator = ConversionOrchestrator(CONFIG_PATH)
    return orchestrator.execute_conversion()

convert_model = PythonOperator(
    task_id='convert_to_torchserve',
    python_callable=convert_model_task,
    dag=dag
)

# Set dependencies
# upload_data >> training_job >> download_model >> 
convert_model
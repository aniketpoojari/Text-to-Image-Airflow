FROM apache/airflow:2.7.1-python3.10 as dependencies

USER root

COPY requirements.txt /requirements.txt

USER airflow

# Pin PyTorch to exactly 2.0.0 (highest SageMaker supported version)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.0.0 \
    torchvision==0.15.1 \
    transformers==4.33.2 \
    sagemaker==2.178.0 \
    platformdirs==3.8.1 \
    protobuf==4.21.12 \
    "botocore>=1.31.85,<1.32.0" \
    "boto3>=1.28.0,<1.29.0"

FROM dependencies as final

RUN pip install --no-cache-dir -r /requirements.txt

RUN pip install --no-cache-dir diffusers
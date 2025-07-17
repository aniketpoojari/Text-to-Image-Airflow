# 🧠 Diffusion-Based Text-to-Image Generation

A robust, production-ready Generative AI system for **generating realistic images from text prompts using diffusion models**. This platform enables end-to-end automation—covering dataset preparation, fully distributed training, validation, experiment tracking, and scalable deployment for inference. The architecture prioritizes modularity, reproducibility, and extensible engineering, targeting real-world research and production needs in generative AI.

Below the surface, the pipeline integrates Amazon SageMaker for scalable distributed training across GPU clusters, MLflow for comprehensive experiment and artifacts tracking, Airflow for orchestration of the entire ML lifecycle, ONNX export for high-performance inference (including TorchServe compatibility), and S3-based data and model management. All processes are fully containerized, with Python best practices (including FastAPI/Pydantic validation and tightly managed dependencies) ensuring maintainability and cloud portability.

## 🚀 Features

- **Advanced Modeling**: Custom diffusion pipeline featuring UNet, variational autoencoder (VAE), and CLIP text encoder for effective text-to-image conditioning.
- **Distributed, Cloud-Native Training**: Supports DeepSpeed-powered training on AWS SageMaker for rapid, elastic scaling across multiple GPUs in the cloud.
- **Centralized Experiment Tracking**: MLflow on DagsHub integration enables centralized, automatic logging of experiments, hyperparameters, model checkpoints, and metrics for complete reproducibility.
- **Seamless Deployment**: Exports trained models to ONNX format for efficient, hardware-agnostic inference. TorchServe integration delivers production-ready API endpoints.
- **End-to-End Automation & Orchestration**: Apache Airflow DAGs manage data uploads to S3 -> SageMaker distributed training -> Model download using DagsHub and S3 -> ONNX export, ensuring a full ML lifecycle.


## 🔧 Quickstart

```bash
docker-compose build --parallel   # builds Docker images
docker-compose up -d              # launches Airflow and dependencies
# Update configs (config/pipeline_config.yaml), place your text-image data, update .env
# Go to http://localhost:8080 to view Airflow UI
# Search for "ml_training_dag" to start the pipeline
```

## 📌 Results

| Metric              | Value          |
|---------------------|----------------|
| Validation VAE Loss | 0.0268         |
| Diffuser Loss       | 0.0241         |
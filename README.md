# 🧠 Diffusion-Based Text-to-Image Generation

This robust, production-ready Generative AI platform specializes in text-to-image generation powered by diffusion models. By utilizing advanced model architectures such as UNet, variational autoencoders (VAEs), and CLIP text encoders, the system precisely translates natural language prompts into realistic and high-quality images. These modeling components work together to enable controllable image synthesis, supporting both cutting-edge research and real-world deployment scenarios.

The platform’s core machine learning workflow leverages Amazon SageMaker for scalable distributed training across GPU clusters, ensuring rapid experimentation and cost-effective development. MLflow manages every aspect of experiment tracking—logging hyperparameters, model checkpoints, and performance metrics for full reproducibility. Apache Airflow orchestrates the entire lifecycle, automating data preparation, model training, artifact management, and deployment. For seamless inference, models are exported to ONNX format, making them compatible with systems like TorchServe. Strict configuration validation and environment management is achieved using Pydantic-backed YAML files, providing schema enforcement, reliable deployments, and simplified experimentation.

## 🚀 Features

- **Advanced Modeling**: Integrates a state-of-the-art diffusion pipeline featuring UNet for noise prediction at time step t with text-to-image guidance, variational autoencoder (VAE) for latent space encoding, and CLIP text encoder for effective text-to-image conditioning.
- **Distributed, Cloud-Native Training**: DeepSpeed-powered distributed training on AWS SageMaker for rapid, elastic scaling across multiple GPUs in the cloud.
- **Centralized Experiment Tracking**: MLflow on DagsHub integration enables centralized, automatic logging of experiments, hyperparameters, model checkpoints, and metrics for complete reproducibility.
- **Seamless Deployment**: Exports trained models to ONNX format for efficient, hardware-agnostic inference. TorchServe integration delivers production-ready API endpoints.
- **End-to-End Automation & Orchestration**: Apache Airflow DAGs manage data uploads to S3 -> SageMaker distributed training -> Model download using DagsHub and S3 -> ONNX export, ensuring a full ML lifecycle.
- **Scalable, Modular Architecture**: The pipeline is designed for extensibility and maintainability, with a modular architecture that enables easy integration of new components. DAG -> Orchestration -> Utilities. DAG performs each step of the pipeline, Orchestrators collect data from config and executes utility functions, Utilities perform the actual work.


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
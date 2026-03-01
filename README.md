---
title: Text to Image Generator
emoji: 🌸
colorFrom: pink
colorTo: purple
sdk: streamlit
app_file: app.py
pinned: false
---

# 🧠 Diffusion-Based Text-to-Image Generation

[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-Space-orange)](https://huggingface.co/spaces/aniketp2009gmail/text-to-image-flowers)

A latent diffusion pipeline built from scratch to generate realistic images from text prompts. It uses three core components: a **CLIP text encoder** to understand the prompt, a **Variational Autoencoder (VAE)** to compress images into a latent space, and a **Cross-Attentional UNet** to iteratively remove noise and form the final image.

The system is trained on 8K image-caption pairs using DeepSpeed distributed training on AWS SageMaker. The entire workflow is automated via Apache Airflow, with MLflow tracking experiments and hyperparameters.

**🚀 Try it live on Hugging Face Spaces: [Text-to-Image Flowers Generator](https://huggingface.co/spaces/aniketp2009gmail/text-to-image-flowers)**

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
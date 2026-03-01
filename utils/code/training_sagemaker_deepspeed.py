import os
import json
import copy
import mlflow
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataloader import TextImageDataLoader
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
import deepspeed
import zipfile
import time
import boto3


def unzip_data(rank: int):
    """Extract training data on the SageMaker instance (rank-0 logs only)."""
    input_dir = "/opt/ml/input/data/train"
    data_zip = os.path.join(input_dir, "flowers.zip")

    if not os.path.exists(data_zip):
        if rank == 0:
            print("No zip file found — using raw input directory.")
        return

    if rank == 0:
        print(f"Extracting {data_zip} ...")
    os.makedirs(input_dir, exist_ok=True)
    with zipfile.ZipFile(data_zip, 'r') as zf:
        zf.extractall(input_dir)
    if rank == 0:
        print(f"Extraction complete. Contents: {os.listdir(input_dir)}")


def setup_distributed():
    """Initialise DeepSpeed distributed training from SageMaker env variables."""
    sm_hosts = json.loads(os.environ['SM_HOSTS'])
    sm_current_host = os.environ['SM_CURRENT_HOST']
    world_size = len(sm_hosts)
    rank = sm_hosts.index(sm_current_host)
    local_rank = 0

    os.environ.update({
        'WORLD_SIZE': str(world_size),
        'RANK': str(rank),
        'LOCAL_RANK': str(local_rank),
        'MASTER_ADDR': sm_hosts[0],
        'MASTER_PORT': '29500',
    })

    deepspeed.init_distributed()
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def training():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # ── Hyperparameters from environment ──────────────────────────────────────
    train_size        = int(os.getenv("TRAIN_SIZE", "300"))
    val_size          = int(os.getenv("VAL_SIZE", "30"))
    vae_image_size    = tuple(map(int, os.getenv("VAE_IMAGE_SIZE", "128,128").split(",")))
    max_length        = int(os.getenv("MAX_LENGTH", "77"))
    batch_size        = int(os.getenv("BATCH_SIZE", "4"))
    T                 = int(os.getenv("T", "1000"))
    beta_schedule     = os.getenv("BETA_SCHEDULE", "squaredcos_cap_v2")
    cfg_dropout_prob  = float(os.getenv("CFG_DROPOUT_PROB", "0.1"))

    unet_image_size      = tuple(map(int, os.getenv("UNET_IMAGE_SIZE", "16,16").split(",")))
    in_channels          = int(os.getenv("IN_CHANNELS", "4"))
    out_channels         = int(os.getenv("OUT_CHANNELS", "4"))
    down_block_types     = tuple(os.getenv("DOWN_BLOCK_TYPES").split(","))
    up_block_types       = tuple(os.getenv("UP_BLOCK_TYPES").split(","))
    mid_block_type       = os.getenv("MID_BLOCK_TYPE")
    block_out_channels   = tuple(map(int, os.getenv("BLOCK_OUT_CHANNELS").split(",")))
    layers_per_block     = int(os.getenv("LAYERS_PER_BLOCK"))
    norm_num_groups      = int(os.getenv("NORM_NUM_GROUPS"))
    cross_attention_dim  = int(os.getenv("CROSS_ATTENTION_DIM"))
    attention_head_dim   = int(os.getenv("ATTENTION_HEAD_DIM"))
    dropout              = float(os.getenv("DROPOUT"))
    time_embedding_type  = os.getenv("TIME_EMBEDDING_TYPE")
    act_fn               = os.getenv("ACT_FN")

    unet_learning_rate = float(os.getenv("UNET_LEARNING_RATE"))
    weight_decay       = float(os.getenv("WEIGHT_DECAY"))
    num_epochs         = int(os.getenv("NUM_EPOCHS"))

    s3_mlruns_bucket = os.getenv("S3_MLRUNS_BUCKET")

    # ── MLflow (rank 0 only) ──────────────────────────────────────────────────
    if rank == 0:
        mlflow.set_tracking_uri(os.getenv("SERVER_URI"))
        mlflow.set_experiment(os.getenv("EXPERIMENT_NAME"))
        mlflow.start_run(run_name=os.getenv("RUN_NAME"))
        mlflow.log_params({
            "train_size": train_size, "val_size": val_size,
            "vae_image_size": vae_image_size, "max_length": max_length,
            "batch_size": batch_size, "T": T, "beta_schedule": beta_schedule,
            "cfg_dropout_prob": cfg_dropout_prob,
            "unet_image_size": unet_image_size, "in_channels": in_channels,
            "out_channels": out_channels, "down_block_types": down_block_types,
            "up_block_types": up_block_types, "mid_block_type": mid_block_type,
            "block_out_channels": block_out_channels, "layers_per_block": layers_per_block,
            "norm_num_groups": norm_num_groups, "cross_attention_dim": cross_attention_dim,
            "attention_head_dim": attention_head_dim, "dropout": dropout,
            "time_embedding_type": time_embedding_type, "act_fn": act_fn,
            "unet_learning_rate": unet_learning_rate, "weight_decay": weight_decay,
            "num_epochs": num_epochs, "world_size": world_size,
        })

    unzip_data(rank)

    # ── Dataset & DataLoader ──────────────────────────────────────────────────
    datadir = "/opt/ml/input/data/train/flowers"
    train_dataset = TextImageDataLoader(
        datadir, range=(0, train_size),
        image_size=vae_image_size, max_text_length=max_length, device=str(device),
        training=True,
    )
    val_dataset = TextImageDataLoader(
        datadir, range=(train_size, train_size + val_size),
        image_size=vae_image_size, max_text_length=max_length, device=str(device),
        training=False,
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset,   num_replicas=world_size, rank=rank, shuffle=False)

    # num_workers=0: DeepSpeed calls deepspeed.init_distributed() before the
    # DataLoader is created, which initialises CUDA.  On Linux (SageMaker),
    # the default fork-based multiprocessing then copies the CUDA context into
    # child processes, causing deadlocks.  With pre-computed embeddings the
    # hot path is only image loading, so num_workers=0 has negligible cost.
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler,
    )

    # ── Frozen pretrained VAE ─────────────────────────────────────────────────
    # The VAE is not fine-tuned: stabilityai/sd-vae-ft-mse already produces
    # excellent latents.  Freezing it roughly halves training compute and
    # avoids degrading the pretrained encoder/decoder.
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    for param in vae.parameters():
        param.requires_grad_(False)

    # ── Noise scheduler ───────────────────────────────────────────────────────
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=T, beta_start=1e-4, beta_end=0.02,
        beta_schedule=beta_schedule,
    )

    # ── UNet ──────────────────────────────────────────────────────────────────
    model_diffuser = UNet2DConditionModel(
        sample_size=unet_image_size,
        in_channels=in_channels, out_channels=out_channels,
        down_block_types=down_block_types, up_block_types=up_block_types,
        block_out_channels=block_out_channels, layers_per_block=layers_per_block,
        cross_attention_dim=cross_attention_dim, attention_head_dim=attention_head_dim,
        dropout=dropout, norm_num_groups=norm_num_groups,
        mid_block_type=mid_block_type, time_embedding_type=time_embedding_type,
        act_fn=act_fn,
    )

    # ZeRO-2 without CPU offloading: model parameters fit comfortably on
    # ml.g4dn.xlarge (16 GB).  CPU offloading adds PCIe transfer overhead
    # with no benefit for models this size.
    ds_config = {
        "train_batch_size": batch_size * world_size,
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": unet_learning_rate, "weight_decay": weight_decay},
        },
        "fp16": {
            "enabled": True,
            "auto_cast": True,
            "initial_scale_power": 16,
            "loss_scale_window": 1000,
            "hysteresis": 4,
            "min_loss_scale": 1,
        },
        "zero_optimization": {"stage": 2},
        "gradient_clipping": 0.5,
    }

    model_diffuser, _, _, _ = deepspeed.initialize(model=model_diffuser, config=ds_config)

    if rank == 0:
        print(f"\n=== Training | epochs={num_epochs} | T={T} | beta={beta_schedule} | CFG p={cfg_dropout_prob} ===\n")

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
        model_diffuser.train()
        total_diff_loss = 0.0

        for images, captions in train_loader:
            images  = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)

            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * 0.18215

            # CFG dropout: ~cfg_dropout_prob of batches train unconditionally
            # so the model learns the score difference used at inference time
            if torch.rand(1).item() < cfg_dropout_prob:
                captions = train_dataset.get_null_embedding(latents.shape[0]).to(device)

            ts = torch.randint(0, T, (latents.shape[0],), device=device)
            epsilons = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, epsilons, ts)

            with torch.amp.autocast('cuda'):
                noise_pred = model_diffuser(
                    noisy_latents, ts, encoder_hidden_states=captions, return_dict=False
                )[0]
                diff_loss = F.mse_loss(noise_pred, epsilons)

            model_diffuser.backward(diff_loss)
            model_diffuser.step()
            total_diff_loss += diff_loss.item()

        # Aggregate train loss across ranks
        avg_train = torch.tensor([total_diff_loss / len(train_loader)], device=device)
        torch.distributed.all_reduce(avg_train, op=torch.distributed.ReduceOp.SUM)
        avg_train /= world_size

        # ── Validate ──────────────────────────────────────────────────────────
        model_diffuser.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for images, captions in val_loader:
                images  = images.to(device, non_blocking=True)
                captions = captions.to(device, non_blocking=True)

                latents = vae.encode(images).latent_dist.sample() * 0.18215
                epsilons = torch.randn_like(latents)
                ts = torch.randint(0, T, (latents.shape[0],), device=device)
                noisy_latents = noise_scheduler.add_noise(latents, epsilons, ts)

                with torch.amp.autocast('cuda'):
                    noise_pred = model_diffuser(
                        noisy_latents, ts, encoder_hidden_states=captions, return_dict=False
                    )[0]
                    diff_loss = F.mse_loss(noise_pred, epsilons)

                total_val_loss += diff_loss.item()

        avg_val = torch.tensor([total_val_loss / len(val_loader)], device=device)
        torch.distributed.all_reduce(avg_val, op=torch.distributed.ReduceOp.SUM)
        avg_val /= world_size

        elapsed = time.time() - t0

        if rank == 0:
            mlflow.log_metric("train_diffuser_loss", avg_train[0].item(), step=epoch)
            mlflow.log_metric("val_diffuser_loss",   avg_val[0].item(),   step=epoch)
            print(
                f"[Epoch {epoch+1}/{num_epochs}] "
                f"Train: {avg_train[0].item():.4f} | "
                f"Val: {avg_val[0].item():.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Save best checkpoint
            if avg_val[0].item() < best_val_loss:
                best_val_loss = avg_val[0].item()
                torch.save(model_diffuser.module.state_dict(), 'best_diffuser_state.pth')
                print(f"  -> New best val loss: {best_val_loss:.4f} — checkpoint saved.")

    # ── Save and upload best model ────────────────────────────────────────────
    if rank == 0:
        print("\n=== Saving best model and uploading to S3... ===\n")

        diffuser = UNet2DConditionModel(
            sample_size=unet_image_size,
            in_channels=in_channels, out_channels=out_channels,
            down_block_types=down_block_types, up_block_types=up_block_types,
            block_out_channels=block_out_channels, layers_per_block=layers_per_block,
            cross_attention_dim=cross_attention_dim, attention_head_dim=attention_head_dim,
            dropout=dropout, norm_num_groups=norm_num_groups,
            mid_block_type=mid_block_type, time_embedding_type=time_embedding_type,
            act_fn=act_fn,
        )
        diffuser.load_state_dict(torch.load('best_diffuser_state.pth', map_location='cpu'))
        torch.save(diffuser, 'diffuser.pth')

        run_id = mlflow.active_run().info.run_id
        s3 = boto3.client('s3')
        s3.upload_file('diffuser.pth', s3_mlruns_bucket, f'{run_id}/diffuser.pth')
        print(f"diffuser.pth uploaded to s3://{s3_mlruns_bucket}/{run_id}/diffuser.pth")

        mlflow.end_run()

    deepspeed.comm.destroy_process_group()


if __name__ == "__main__":
    training()

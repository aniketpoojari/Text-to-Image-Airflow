"""
Text-to-Image Inference Demo
-----------------------------
Run:  streamlit run app.py

Models are downloaded automatically from HuggingFace Hub on first launch.
Set the HF_MODEL_REPO environment variable (or HF Spaces variable) to your
model repository, e.g.:  HF_MODEL_REPO=your-username/text-to-image-flowers

Three inference backends are supported:
  PyTorch  — full-precision, works on CPU and CUDA
  ONNX     — optimised for deployment, CPU and CUDA (default for HF Spaces)
  TensorRT — fastest on NVIDIA GPU; requires tensorrt + CUDA (local use)
"""

import os
import yaml
import torch
import numpy as np
import streamlit as st
import onnxruntime as ort
from diffusers import DDPMScheduler, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
import time

# ── Optional TensorRT support ─────────────────────────────────────────────────
try:
    import tensorrt  # noqa: F401  (only needed to confirm TRT is installed)
    _TRT_EP = "TensorrtExecutionProvider"
    TRT_AVAILABLE = _TRT_EP in ort.get_available_providers()
except ImportError:
    TRT_AVAILABLE = False

# ── Load pipeline config so the app always matches training settings ───────────
with open("config/pipeline_config.yaml") as f:
    _cfg = yaml.safe_load(f)

IMAGE_SIZE    = tuple(map(int, _cfg["vae"]["image_size"].split(",")))   # e.g. (128, 128)
LATENT_H      = IMAGE_SIZE[0] // 8
LATENT_W      = IMAGE_SIZE[1] // 8
MAX_LENGTH    = _cfg["clip"]["max_length"]
T             = _cfg["ddpm_scheduler"]["T"]
BETA_SCHEDULE = _cfg["ddpm_scheduler"]["beta_schedule"]
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "")

if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True


# ── HuggingFace Hub model download ────────────────────────────────────────────

@st.cache_resource(show_spinner="Downloading models from HuggingFace Hub...")
def ensure_models():
    """Download ONNX + PyTorch models from HF Hub if not already on disk.

    Set the HF_MODEL_REPO environment variable to your HF Hub model repo.
    Skip silently if the variable is unset (models assumed to be local).
    """
    unet_onnx = "models/onnx_models/unet.onnx"
    if HF_MODEL_REPO and not os.path.exists(unet_onnx):
        from huggingface_hub import snapshot_download
        st.info(f"Downloading models from {HF_MODEL_REPO} ...")
        snapshot_download(repo_id=HF_MODEL_REPO, local_dir="models", repo_type="model")
        st.success("Models downloaded.")


# ── Early-stopping / adaptive CFG helpers ─────────────────────────────────────

def should_stop_early(latents, prev_latents, step, total_steps, threshold=0.08):
    if prev_latents is None or step < total_steps * 0.3:
        return False
    change = torch.mean(torch.abs(latents - prev_latents)).item()
    magnitude = torch.mean(torch.abs(latents)).item()
    return (change / (magnitude + 1e-8)) < threshold


def get_adaptive_cfg(latents, prev_latents, base_cfg=7.5):
    if prev_latents is None:
        return base_cfg
    change = torch.mean(torch.abs(latents - prev_latents)).item()
    if change < 0.05:
        return max(base_cfg * 0.2, 1.0)
    if change < 0.15:
        return base_cfg * 0.6
    return base_cfg


# ── Text embedding helpers ────────────────────────────────────────────────────

def get_text_embeddings_pytorch(prompt, tokenizer, text_encoder):
    device = torch.device(DEVICE)
    def encode(text):
        tokens = tokenizer(text, return_tensors='pt', padding='max_length',
                           truncation=True, max_length=MAX_LENGTH).to(device)
        with torch.no_grad():
            return text_encoder(**tokens).last_hidden_state
    return encode(""), encode(prompt)  # uncond, cond


def get_text_embeddings_onnx(prompt, tokenizer, clip_sess):
    def encode(text):
        inputs = tokenizer(text, return_tensors="np", padding="max_length",
                           truncation=True, max_length=MAX_LENGTH)
        return clip_sess.run(["last_hidden_state"], {
            "input_ids":      inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        })[0]
    device = torch.device(DEVICE)
    uncond_np = encode("")
    cond_np   = encode(prompt)
    return torch.from_numpy(uncond_np).to(device), torch.from_numpy(cond_np).to(device)


# ── Denoising loop ────────────────────────────────────────────────────────────

def denoise(latents, scheduler, uncond_emb, cond_emb, cfg_scale,
            unet=None, unet_sess=None, vae=None, vae_sess=None):
    device = torch.device(DEVICE)
    progress_bar = st.progress(0)
    status_text  = st.empty()
    preview_slot = st.empty()
    prev_latents = None
    total_steps  = len(scheduler.timesteps)

    with torch.no_grad():
        for i, t in enumerate(scheduler.timesteps):
            current_cfg = get_adaptive_cfg(latents, prev_latents, cfg_scale)
            lat_in  = torch.cat([latents, latents])
            txt_emb = torch.cat([uncond_emb, cond_emb])

            if unet:
                noise = unet(lat_in, t.expand(2).to(device),
                             encoder_hidden_states=txt_emb, return_dict=False)[0]
            else:
                # Shared path for both ONNX and TensorRT (ORT TRT execution provider)
                t_batch = torch.full((2,), t.item() if isinstance(t, torch.Tensor) else t,
                                     dtype=torch.float32, device=device)
                noise_np = unet_sess.run(["noise_pred"], {
                    "latents":               lat_in.cpu().numpy(),
                    "timesteps":             t_batch.cpu().numpy(),
                    "encoder_hidden_states": txt_emb.cpu().numpy(),
                })[0]
                noise = torch.from_numpy(noise_np).to(device)

            n_uncond, n_cond = noise.chunk(2)
            noise_pred = n_uncond + current_cfg * (n_cond - n_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

            if should_stop_early(latents, prev_latents, i, total_steps):
                status_text.write(f"Early stop at step {i+1}/{total_steps} — converged.")
                break

            prev_latents = latents.clone()
            progress_bar.progress((i + 1) / total_steps)
            status_text.write(f"Step {i+1}/{total_steps}  (Auto-CFG: {current_cfg:.1f})")

            # Live preview every 5 steps
            if i % 5 == 0:
                try:
                    preview = latents / 0.18215
                    if vae:
                        img = vae.decode(preview).sample[0].permute(1, 2, 0).clamp(-1, 1).cpu().numpy()
                        img = (img + 1.0) / 2.0
                    else:
                        img_np = vae_sess.run(["image"], {"latents": preview.cpu().numpy()})[0]
                        img = np.clip(img_np[0].transpose(1, 2, 0), 0, 1)
                    preview_slot.image(img, caption=f"Step {i+1}", width=IMAGE_SIZE[0] * 2)
                except Exception:
                    pass

    progress_bar.empty()
    status_text.empty()
    return latents


# ── Model loading (cached) ────────────────────────────────────────────────────

@st.cache_resource
def load_pytorch_models():
    device = DEVICE
    vae          = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    unet         = torch.load("models/diffuser.pth", map_location=device, weights_only=False).eval()
    tokenizer    = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    scheduler    = DDPMScheduler(num_train_timesteps=T, beta_start=1e-4,
                                 beta_end=0.02, beta_schedule=BETA_SCHEDULE)
    return vae, unet, tokenizer, text_encoder, scheduler


def _onnx_providers(cuda=True):
    if cuda and DEVICE == "cuda":
        return [('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }), 'CPUExecutionProvider']
    return ['CPUExecutionProvider']


@st.cache_resource
def load_onnx_models():
    providers = _onnx_providers(cuda=True)
    clip_sess = ort.InferenceSession("models/onnx_models/clip_text_encoder.onnx", providers=providers)
    unet_sess = ort.InferenceSession("models/onnx_models/unet.onnx",              providers=providers)
    vae_sess  = ort.InferenceSession("models/onnx_models/vae_decoder.onnx",       providers=providers)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    scheduler = DDPMScheduler(num_train_timesteps=T, beta_start=1e-4,
                              beta_end=0.02, beta_schedule=BETA_SCHEDULE)
    return clip_sess, unet_sess, vae_sess, tokenizer, scheduler


@st.cache_resource
def load_trt_models():
    """Load ONNX models with TensorRT execution provider (caches engines to disk).

    Engines are built on first call (~1-2 min) and cached to models/trt_cache/.
    Subsequent calls load the cached engines instantly.
    """
    trt_cache = "models/trt_cache"
    os.makedirs(trt_cache, exist_ok=True)
    providers = [
        ('TensorrtExecutionProvider', {
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': trt_cache,
            'trt_fp16_enable': True,
            'device_id': 0,
        }),
        ('CUDAExecutionProvider', {'device_id': 0}),
    ]
    clip_sess = ort.InferenceSession("models/onnx_models/clip_text_encoder.onnx", providers=providers)
    unet_sess = ort.InferenceSession("models/onnx_models/unet.onnx",              providers=providers)
    vae_sess  = ort.InferenceSession("models/onnx_models/vae_decoder.onnx",       providers=providers)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    scheduler = DDPMScheduler(num_train_timesteps=T, beta_start=1e-4,
                              beta_end=0.02, beta_schedule=BETA_SCHEDULE)
    return clip_sess, unet_sess, vae_sess, tokenizer, scheduler


# ── Image generation ──────────────────────────────────────────────────────────

def generate(prompt, cfg_scale, steps, method):
    device = torch.device(DEVICE)
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    if method == "PyTorch":
        vae, unet, tokenizer, text_encoder, scheduler = load_pytorch_models()
        uncond_emb, cond_emb = get_text_embeddings_pytorch(prompt, tokenizer, text_encoder)
        latents = torch.randn((1, 4, LATENT_H, LATENT_W), device=device)
        scheduler.set_timesteps(steps)
        latents *= scheduler.init_noise_sigma
        latents = denoise(latents, scheduler, uncond_emb, cond_emb, cfg_scale,
                          unet=unet, vae=vae)
        latents /= 0.18215
        with torch.no_grad():
            img = vae.decode(latents).sample[0].permute(1, 2, 0).clamp(-1, 1).cpu().numpy()
        return (img + 1.0) / 2.0

    elif method == "TensorRT":
        clip_sess, unet_sess, vae_sess, tokenizer, scheduler = load_trt_models()
        uncond_emb, cond_emb = get_text_embeddings_onnx(prompt, tokenizer, clip_sess)
        latents = torch.randn((1, 4, LATENT_H, LATENT_W), device=device, dtype=torch.float32)
        scheduler.set_timesteps(steps)
        sigma = scheduler.init_noise_sigma
        latents *= sigma.item() if isinstance(sigma, torch.Tensor) else sigma
        latents = denoise(latents, scheduler, uncond_emb, cond_emb, cfg_scale,
                          unet_sess=unet_sess, vae_sess=vae_sess)
        latents /= 0.18215
        img_np = vae_sess.run(["image"], {"latents": latents.cpu().numpy()})[0]
        return np.clip(img_np[0].transpose(1, 2, 0), 0, 1)

    else:  # ONNX
        clip_sess, unet_sess, vae_sess, tokenizer, scheduler = load_onnx_models()
        uncond_emb, cond_emb = get_text_embeddings_onnx(prompt, tokenizer, clip_sess)
        latents = torch.randn((1, 4, LATENT_H, LATENT_W), device=device, dtype=torch.float32)
        scheduler.set_timesteps(steps)
        sigma = scheduler.init_noise_sigma
        latents *= sigma.item() if isinstance(sigma, torch.Tensor) else sigma
        latents = denoise(latents, scheduler, uncond_emb, cond_emb, cfg_scale,
                          unet_sess=unet_sess, vae_sess=vae_sess)
        latents /= 0.18215
        img_np = vae_sess.run(["image"], {"latents": latents.cpu().numpy()})[0]
        return np.clip(img_np[0].transpose(1, 2, 0), 0, 1)


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.title("Text-to-Image Generator")
st.caption(f"Model config: T={T}, beta_schedule={BETA_SCHEDULE}, image_size={IMAGE_SIZE}")

# Download models from HF Hub if needed (no-op if already on disk or HF_MODEL_REPO not set)
ensure_models()

# Build the list of available backends
_backends = ["ONNX", "PyTorch"]
if TRT_AVAILABLE and DEVICE == "cuda":
    _backends.insert(0, "TensorRT")

with st.sidebar:
    st.header("Settings")
    method    = st.selectbox("Inference backend", _backends,
                              help="TensorRT: fastest on NVIDIA GPU (local). ONNX: recommended for HF Spaces.")
    cfg_scale = st.slider("CFG scale", 1.0, 20.0, 7.5, 0.5,
                          help="Higher = more prompt-adherent, lower = more creative")
    steps     = st.slider("Denoising steps", 10, 300, 50, 10)
    st.markdown("---")
    st.markdown(f"**Device:** {DEVICE.upper()}")
    st.markdown(f"**Image size:** {IMAGE_SIZE[0]}×{IMAGE_SIZE[1]}")
    if method == "TensorRT":
        st.info("TRT engines are built on first use (~1-2 min) and cached to disk.")

with st.spinner("Loading models..."):
    if method == "PyTorch":
        load_pytorch_models()
    elif method == "TensorRT":
        load_trt_models()
    else:
        load_onnx_models()

prompt = st.text_input("Prompt", placeholder="A red rose with morning dew drops")

if st.button("Generate", type="primary") and prompt.strip():
    t0 = time.time()
    with st.spinner("Generating..."):
        img = generate(prompt, cfg_scale, steps, method)
    elapsed = time.time() - t0

    if img is not None:
        st.success(f"Generated in {elapsed:.1f}s  ({method})")
        st.image(img, caption=prompt, width=IMAGE_SIZE[0] * 3)
    else:
        st.error("Generation failed.")

"""
TensorRT Converter — local NVIDIA GPU only
==========================================
Converts the ONNX models produced by model_converter.py into optimised
TensorRT engine files for maximum inference speed on NVIDIA hardware.

Requirements (NOT in requirements.txt — CUDA-version-specific):
  pip install tensorrt   # matches your CUDA version, e.g. tensorrt==8.6.1.post1
  pip install pycuda     # for host<->device memory management

Usage:
  python utils/tensorrt_converter.py \
      --onnx_dir  models/onnx_models \
      --trt_dir   models/trt_models  \
      --fp16                          # enable FP16 (faster, minimal quality loss)

The produced .trt engine files are device-specific: they are built for the
exact GPU architecture on which this script is run.  Re-run this script if
you switch to a different GPU model.
"""

import os
import sys
import argparse
from typing import Optional


def _require_tensorrt():
    """Import tensorrt and raise a helpful error if it is not installed."""
    try:
        import tensorrt as trt
        return trt
    except ImportError:
        print(
            "\n[ERROR] tensorrt is not installed.\n"
            "Install it with the command matching your CUDA version, e.g.:\n"
            "  pip install tensorrt==8.6.1.post1   # CUDA 11.x\n"
            "  pip install tensorrt==10.0.1         # CUDA 12.x\n"
            "See https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html\n",
            file=sys.stderr,
        )
        sys.exit(1)


class TensorRTConverter:
    """Converts ONNX diffusion model components to TensorRT engine files.

    Only the three models used by the inference pipeline are converted:
      - clip_text_encoder.onnx → clip_text_encoder.trt
      - unet.onnx               → unet.trt
      - vae_decoder.onnx        → vae_decoder.trt

    FP16 mode is recommended: it roughly doubles throughput on Ampere/Turing
    GPUs with negligible impact on generated image quality.
    """

    MODELS = [
        "clip_text_encoder.onnx",
        "unet.onnx",
        "vae_decoder.onnx",
    ]

    def convert_from_onnx(
        self,
        onnx_dir: str,
        trt_dir: str,
        fp16: bool = True,
        workspace_gb: int = 4,
    ) -> dict:
        """Build TensorRT engines from ONNX files.

        Args:
            onnx_dir:     Directory containing the three .onnx files.
            trt_dir:      Output directory for .trt engine files.
            fp16:         Enable FP16 precision (requires Volta+ GPU).
            workspace_gb: GPU workspace size in GB for the TRT builder.

        Returns:
            dict with keys: converted, skipped, failed, output_dir.
        """
        trt = _require_tensorrt()

        os.makedirs(trt_dir, exist_ok=True)

        logger = trt.Logger(trt.Logger.WARNING)
        converted, skipped, failed = [], [], []

        for model_file in self.MODELS:
            onnx_path = os.path.join(onnx_dir, model_file)
            trt_path = os.path.join(trt_dir, model_file.replace(".onnx", ".trt"))

            if not os.path.exists(onnx_path):
                print(f"  [SKIP] {model_file} — ONNX file not found at {onnx_path}")
                skipped.append(model_file)
                continue

            if os.path.exists(trt_path):
                print(f"  [SKIP] {model_file} — TRT engine already exists at {trt_path}")
                skipped.append(model_file)
                continue

            print(f"  [BUILD] {model_file} → {trt_path} ...")
            success = self._build_engine(
                trt, logger, onnx_path, trt_path, fp16=fp16, workspace_gb=workspace_gb
            )
            if success:
                size_mb = os.path.getsize(trt_path) / (1024 ** 2)
                print(f"    Done — {size_mb:.1f} MB")
                converted.append(model_file)
            else:
                print(f"    [FAILED] Could not build engine for {model_file}")
                failed.append(model_file)

        return {
            "converted": converted,
            "skipped": skipped,
            "failed": failed,
            "output_dir": trt_dir,
        }

    def _build_engine(
        self,
        trt,
        logger,
        onnx_path: str,
        engine_path: str,
        fp16: bool,
        workspace_gb: int,
    ) -> bool:
        """Parse an ONNX model and serialise the TensorRT engine to disk."""
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"    ONNX parse error: {parser.get_error(i)}")
                return False

        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30)
        )

        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("    FP16 enabled")
        elif fp16:
            print("    FP16 requested but GPU does not support fast FP16; using FP32")

        # Optimisation profile for dynamic batch axis (dim 0)
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            inp = network.get_input(i)
            shape = list(inp.shape)
            if shape[0] == -1:  # dynamic batch
                shape[0] = 1
                min_shape = tuple(shape)
                opt_shape = tuple([2] + shape[1:])
                max_shape = tuple([4] + shape[1:])
                profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

        serialised = builder.build_serialized_network(network, config)
        if serialised is None:
            return False

        with open(engine_path, "wb") as f:
            f.write(serialised)
        return True

    @staticmethod
    def load_engine(trt, engine_path: str):
        """Load a serialised TensorRT engine from disk.

        Returns a trt.ICudaEngine ready for inference context creation.
        """
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            return runtime.deserialize_cuda_engine(f.read())


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX models to TensorRT engines")
    parser.add_argument("--onnx_dir",    default="models/onnx_models",  help="ONNX model directory")
    parser.add_argument("--trt_dir",     default="models/trt_models",   help="TRT engine output directory")
    parser.add_argument("--fp16",        action="store_true",            help="Enable FP16 precision")
    parser.add_argument("--workspace_gb", type=int, default=4,          help="Builder workspace in GB")
    args = parser.parse_args()

    converter = TensorRTConverter()
    print(f"\nConverting ONNX models → TensorRT engines")
    print(f"  ONNX dir  : {args.onnx_dir}")
    print(f"  Output dir: {args.trt_dir}")
    print(f"  FP16      : {args.fp16}\n")

    result = converter.convert_from_onnx(
        onnx_dir=args.onnx_dir,
        trt_dir=args.trt_dir,
        fp16=args.fp16,
        workspace_gb=args.workspace_gb,
    )

    print(f"\nDone.")
    print(f"  Converted : {result['converted']}")
    print(f"  Skipped   : {result['skipped']}")
    print(f"  Failed    : {result['failed']}")

"""Record numerical runtime settings separately from parameter storage dtypes."""
import os

import torch


def precision_snapshot():
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_bf16_reduced_precision_reduction":
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "cuda_matmul_allow_fp16_reduced_precision_reduction":
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
    }


def configure_matmul_precision(requested):
    """Call after model construction: OpenPI's constructor sets precision to high."""
    if requested not in ("upstream", "highest", "high", "medium"):
        raise ValueError(f"Unknown float32 matmul precision: {requested}")
    if requested != "upstream":
        torch.set_float32_matmul_precision(requested)
        if torch.get_float32_matmul_precision() != requested:
            raise RuntimeError("Requested float32 matmul precision did not take effect")
    return precision_snapshot()

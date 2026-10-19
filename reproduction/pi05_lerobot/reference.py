#!/usr/bin/env python3
"""User-run, offline, one-sample reference probe. Never trains or quantizes."""
import argparse
import hashlib
import json
import os
from pathlib import Path

for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[name] = "1"
from compat import (load_config, make_preprocessor, synthetic_sample, calib_sample,
                    validate_preprocessing, load_strict, openpi_inputs_from_reference, validate_tokenizer)
ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("inspect", "preprocess", "forward"), default="inspect")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "models/pi05_libero")
    p.add_argument("--tokenizer", type=Path, default=ROOT / "models/paligemma_tokenizer")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--synthetic", action="store_true", help="Interface check only, not a quality metric")
    group.add_argument("--calib-dir", type=Path)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--output", type=Path, help="New directory; existing paths are rejected")
    args = p.parse_args()
    cfg = load_config(args.checkpoint)
    report = {"mode": args.mode, "chunk_size": cfg.chunk_size,
              "max_action_dim": cfg.max_action_dim, "tokenizer_max_length": cfg.tokenizer_max_length,
              "normalization": {k:v.value for k,v in cfg.normalization_mapping.items()},
              "weights_present": (args.checkpoint / "model.safetensors").is_file()}
    if args.mode == "inspect":
        print(json.dumps(report, ensure_ascii=False, indent=2)); return
    if not args.synthetic and args.calib_dir is None:
        p.error("Choose --synthetic or --calib-dir explicitly")
    if args.output is None:
        p.error("--output must name a new directory")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if not (args.tokenizer / "tokenizer_config.json").is_file():
        raise FileNotFoundError("Tokenizer assets missing; run download_assets.py --tokenizer yourself first")
    if args.mode == "forward" and not (args.checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError("Weights missing; run download_assets.py --weights yourself first")
    import numpy as np
    import torch
    raw = synthetic_sample(cfg) if args.synthetic else calib_sample(args.calib_dir, args.index, cfg)
    report["raw_image_shapes"] = {k: list(v.shape) for k, v in raw.items()
                                  if k.startswith("observation.images.")}
    preprocessor = make_preprocessor(args.checkpoint, args.tokenizer, "cpu")
    # Keep independent raw copies because processors may mutate dictionaries.
    batch = preprocessor({k:v.clone() if isinstance(v,torch.Tensor) else list(v) for k,v in raw.items()})
    report.update(validate_preprocessing(raw, batch, args.checkpoint, cfg))
    report["sample_kind"] = "synthetic_interface_only" if args.synthetic else "raw_calibration"
    report["calib_dir"] = str(args.calib_dir) if args.calib_dir else None
    report["index"] = args.index
    report.update(validate_tokenizer(batch, args.checkpoint, args.tokenizer))
    arrays = {k:v.cpu().numpy() for k,v in batch.items() if isinstance(v,torch.Tensor)}
    if args.mode == "forward":
        torch.manual_seed(42)
        policy = load_strict(args.checkpoint, args.device)
        from runtime_precision import precision_snapshot
        report["runtime_precision_before_forward"] = precision_snapshot()
        batch = {k:v.to(args.device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
        obs, actions = openpi_inputs_from_reference(policy, batch)
        report["model_image_shapes"] = {k: list(v.shape) for k, v in obs.images.items()}
        noise = torch.randn(actions.shape, generator=torch.Generator().manual_seed(42)).to(args.device)
        time = torch.tensor([0.5], device=args.device)
        with torch.inference_mode():
            losses = policy.model(list(obs.images.values()), list(obs.image_masks.values()),
                                  obs.tokenized_prompt, obs.tokenized_prompt_mask, actions,
                                  noise=noise, time=time)
        if tuple(losses.shape) != (1, cfg.chunk_size, cfg.max_action_dim) or not torch.isfinite(losses).all():
            raise RuntimeError("Reference forward returned invalid shape or non-finite loss")
        report.update(strict_load="PASS", forward="PASS", loss_shape=list(losses.shape),
                      loss_7d=float(losses[:,:,:7].mean()), loss_32d=float(losses.mean()),
                      seed=42, time=0.5, dtype=cfg.dtype,
                      peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30 if args.device=="cuda" else 0)
        for name,value in obs.images.items(): arrays["images."+name] = value.cpu().numpy()
        for name,value in obs.image_masks.items(): arrays["image_masks."+name] = value.cpu().numpy()
        arrays.update(actions=actions.cpu().numpy(), noise=noise.cpu().numpy(),
                      time=time.cpu().numpy(), per_element_loss=losses.cpu().numpy())
    report["config_sha256"] = hashlib.sha256((args.checkpoint / "config.json").read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez(args.output / "inputs_and_reference.npz", **arrays)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Saved:", args.output)
    if args.synthetic:
        print("A synthetic PASS checks interfaces only; it is not LIBERO success or quantization validation.")
    else:
        print("A real-sample PASS checks this input path only; semantic alignment, quantization and LIBERO remain unvalidated.")


if __name__ == "__main__":
    main()

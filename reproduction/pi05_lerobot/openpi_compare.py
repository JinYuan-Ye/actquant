#!/usr/bin/env python3
"""Inspect or replay a saved LeRobot probe in isolated OpenPI. No calibration/backward."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import struct
from types import SimpleNamespace

for _name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[_name] = "1"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import torch
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from runtime_precision import configure_matmul_precision, precision_snapshot

ROOT = Path(__file__).resolve().parents[2]
CAMERAS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
DTYPES = {torch.float32: "F32", torch.bfloat16: "BF16"}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_header(path):
    with path.open("rb") as stream:
        size_bytes = stream.read(8)
        if len(size_bytes) != 8:
            raise ValueError("Truncated safetensors header length")
        size = struct.unpack("<Q", size_bytes)[0]
        if not 0 < size < 16 * 1024 * 1024:
            raise ValueError("Unexpected safetensors header size")
        payload = stream.read(size)
        if len(payload) != size:
            raise ValueError("Truncated safetensors header")
    header = json.loads(payload)
    return {k: v for k, v in header.items() if k != "__metadata__"}, hashlib.sha256(payload).hexdigest()


def strip_policy_prefix(state):
    # The pinned checkpoint already has the target time_mlp and adaRMS names.
    # Do not drop keys, rename unmatched layers, or coerce incompatible shapes.
    if not state or not all(key.startswith("model.") for key in state):
        raise ValueError("Expected the pinned LeRobot checkpoint's model. prefix on every key")
    return {key[len("model."):]: value for key, value in state.items()}


def build_config(checkpoint):
    raw = json.loads((checkpoint / "config.json").read_text())
    if raw.get("type") != "pi05" or raw.get("dtype") != "bfloat16":
        raise ValueError("This comparison expects the pinned BF16 pi05 configuration")
    if (raw["chunk_size"], raw["max_action_dim"], raw["tokenizer_max_length"]) != (50, 32, 200):
        raise ValueError("Expected horizon50/action32/tokens200 for the saved probe")
    if raw["image_resolution"] != [224, 224]:
        raise ValueError("Unexpected model image resolution")
    return Pi0Config(
        pi05=True, action_horizon=raw["chunk_size"], action_dim=raw["max_action_dim"],
        max_token_len=raw["tokenizer_max_length"], dtype=raw["dtype"],
        paligemma_variant=raw["paligemma_variant"],
        action_expert_variant=raw["action_expert_variant"], pytorch_compile_mode=None,
    )


def fill_tied_aliases(model, state):
    groups = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        groups.setdefault(id(param), []).append(name)
    filled = []
    for names in groups.values():
        present = [name for name in names if name in state]
        if present:
            for name in names:
                if name not in state:
                    state[name] = state[present[0]]
                    filled.append([name, present[0]])
    return filled


class SavedInputPI05(PI0Pytorch):
    """Use the official forward, with augmentation disabled for reference replay only.

    Upstream forward hard-codes train=True even in eval mode. This local override
    explicitly chooses the no-augmentation preprocessing path and checks that the
    already preprocessed images, masks, tokens and state remain bitwise equal.
    This class is not wired into Fisher/HSIC and does not define their augmentation policy.
    """
    def _preprocess_observation(self, observation, *, train=True):
        result = super()._preprocess_observation(observation, train=False)
        images, masks, tokens, token_mask, state = result
        expected = [*observation.images.values(), *observation.image_masks.values(),
                    observation.tokenized_prompt, observation.tokenized_prompt_mask, observation.state]
        actual = [*images, *masks, tokens, token_mask, state]
        if len(actual) != len(expected) or any(
            a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b)
            for a, b in zip(actual, expected)
        ):
            raise ValueError("OpenPI preprocessing changed the saved reference input")
        return result


def audit_schema(checkpoint, config):
    source, digest = checkpoint_header(checkpoint / "model.safetensors")
    mapped = strip_policy_prefix(source)
    # Meta tensors record shapes/dtypes but allocate no model weight storage.
    with torch.device("meta"):
        model = SavedInputPI05(config)
    target = model.state_dict()
    if not all(value.is_meta for value in target.values()):
        raise RuntimeError("Schema audit unexpectedly allocated model tensors")
    aliases = fill_tied_aliases(model, mapped)
    missing = sorted(target.keys() - mapped.keys())
    unexpected = sorted(mapped.keys() - target.keys())
    shape_errors, dtype_errors = [], []
    for key in sorted(target.keys() & mapped.keys()):
        if list(target[key].shape) != mapped[key]["shape"]:
            shape_errors.append(key)
        if DTYPES.get(target[key].dtype) != mapped[key]["dtype"]:
            dtype_errors.append(key)
    result = {
        "schema_match": "PASS" if not (missing or unexpected or shape_errors or dtype_errors) else "FAIL",
        "source_tensor_count": len(source), "target_state_key_count": len(target),
        "tied_aliases_filled": aliases, "missing": missing, "unexpected": unexpected,
        "shape_mismatches": shape_errors, "dtype_mismatches": dtype_errors,
        "checkpoint_header_sha256": digest,
        "checkpoint_bytes": (checkpoint / "model.safetensors").stat().st_size,
        "target_state_dtype_counts": dict(Counter(str(v.dtype) for v in target.values())),
        "model_weight_storage_allocated": False,
    }
    return result


def read_reference(directory, checkpoint, config):
    report = json.loads((directory / "report.json").read_text())
    if report.get("forward") != "PASS" or report.get("strict_load") != "PASS":
        raise ValueError("Need a successful LeRobot forward reference")
    if report.get("config_sha256") != file_hash(checkpoint / "config.json"):
        raise ValueError("Reference config hash differs from current checkpoint")
    with np.load(directory / "inputs_and_reference.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    expected_shapes = {
        "actions": (1, config.action_horizon, config.action_dim),
        "noise": (1, config.action_horizon, config.action_dim),
        "per_element_loss": (1, config.action_horizon, config.action_dim),
        "time": (1,), "observation.state": (1, 8), "action": (1, config.action_horizon, 7),
        "observation.language.tokens": (1, config.max_token_len),
        "observation.language.attention_mask": (1, config.max_token_len),
    }
    for camera in CAMERAS:
        expected_shapes["images." + camera] = (1, 3, 224, 224)
        expected_shapes["image_masks." + camera] = (1,)
    for key, shape in expected_shapes.items():
        if key not in arrays or arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid/missing reference tensor: {key}, expected {shape}")
    for key in ("actions", "noise", "time", "observation.state", "action", "per_element_loss"):
        if arrays[key].dtype != np.float32:
            raise ValueError(f"Expected float32 reference tensor: {key}")
    for camera, expected_mask in zip(CAMERAS, (True, True, False)):
        values = arrays["images." + camera]
        mask = arrays["image_masks." + camera]
        if values.dtype != np.float32 or np.min(values) < -1 or np.max(values) > 1:
            raise ValueError(f"Expected preprocessed float32 [-1,1] image: {camera}")
        if mask.dtype != np.bool_ or mask.tolist() != [expected_mask]:
            raise ValueError(f"Unexpected camera mask: {camera}")
    if not np.all(arrays["images.right_wrist_0_rgb"] == -1):
        raise ValueError("Masked placeholder image differs from the official reference")
    tokens = arrays["observation.language.tokens"]
    token_mask = arrays["observation.language.attention_mask"]
    if tokens.dtype.kind not in "iu" or tokens.min() < 0 or tokens.max() >= 257152:
        raise ValueError("Invalid reference token IDs")
    if not np.isin(token_mask, [0, 1]).all() or np.any(np.diff(token_mask.astype(np.int8), axis=1) > 0):
        raise ValueError("Expected a right-padded reference attention mask")
    if int(token_mask.sum()) != report["valid_tokens"]:
        raise ValueError("Reference token mask does not match report")
    if not np.array_equal(arrays["actions"][:, :, :7], arrays["action"]) or np.any(arrays["actions"][:, :, 7:] != 0):
        raise ValueError("Reference action padding is inconsistent")
    if report.get("dtype") != config.dtype or not np.array_equal(arrays["time"], [report["time"]]):
        raise ValueError("Reference time/dtype differs from the report/config")
    loss = arrays["per_element_loss"]
    if np.any(loss < 0):
        raise ValueError("Negative per-element squared loss")
    np.testing.assert_allclose(loss[:, :, :7].mean(), report["loss_7d"], rtol=1e-6)
    np.testing.assert_allclose(loss.mean(), report["loss_32d"], rtol=1e-6)
    return arrays, report


def observation_from_arrays(arrays, device):
    def tensor(key):
        return torch.from_numpy(arrays[key].copy()).to(device)
    return SimpleNamespace(
        images={key: tensor("images." + key) for key in CAMERAS},
        image_masks={key: tensor("image_masks." + key) for key in CAMERAS},
        state=tensor("observation.state"),
        tokenized_prompt=tensor("observation.language.tokens").long(),
        tokenized_prompt_mask=tensor("observation.language.attention_mask").bool(),
        token_ar_mask=None, token_loss_mask=None,
    ), tensor("actions"), tensor("noise"), tensor("time")


def compare_losses(actual, expected, *, atol, rtol):
    if actual.shape != expected.shape or not np.isfinite(actual).all() or np.any(actual < 0):
        raise ValueError("Target returned invalid per-element loss")
    diff = actual.astype(np.float64) - expected.astype(np.float64)
    absolute = np.abs(diff)
    close = absolute <= atol + rtol * np.abs(expected)
    exact = np.array_equal(actual, expected)
    status = "EXACT_MATCH" if exact else ("WITHIN_TOLERANCE" if close.all() else "MISMATCH")
    return {
        "status": status, "atol": atol, "rtol": rtol,
        "max_abs_diff": float(absolute.max()), "mean_abs_diff": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "elements_outside_tolerance": int(np.count_nonzero(~close)), "total_elements": int(actual.size),
        "worst_index": list(np.unravel_index(int(absolute.argmax()), absolute.shape)),
        "max_abs_diff_7d": float(absolute[:, :, :7].max()),
        "max_abs_diff_padded_25d": float(absolute[:, :, 7:].max()),
    }


def load_target(checkpoint, config, device):
    from safetensors.torch import load_file
    model = SavedInputPI05(config)
    state = strip_policy_prefix(load_file(str(checkpoint / "model.safetensors"), device="cpu"))
    aliases = fill_tied_aliases(model, state)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device).eval()  # Preserve the official BF16/FP32 mixture.
    return model, aliases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("inspect", "forward"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/pi05_libero")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--float32-matmul-precision", choices=("upstream", "highest", "high", "medium"),
                        default="upstream", help="Apply after construction; highest matches the uncompiled LeRobot reference setup")
    args = parser.parse_args()
    if not np.isfinite([args.atol, args.rtol]).all() or min(args.atol, args.rtol) < 0:
        parser.error("Tolerances must be finite and non-negative")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    config = build_config(args.checkpoint)
    arrays, reference = read_reference(args.reference, args.checkpoint, config)
    report = {"mode": args.mode, "reference": str(args.reference.resolve()),
              "reference_npz_sha256": file_hash(args.reference / "inputs_and_reference.npz"),
              "reference_report_sha256": file_hash(args.reference / "report.json"),
              "config_sha256": file_hash(args.checkpoint / "config.json"),
              "action_horizon": config.action_horizon, "action_dim": config.action_dim,
              "max_token_len": config.max_token_len, "dtype": config.dtype,
              "float32_matmul_precision_request": args.float32_matmul_precision,
              "reference_input_validation": "PASS", "augmentation": "disabled_for_saved_input_comparison",
              "loss_scope": "per_element_32d; also report first7d mean; no backward",
              "scope": "single_probe_loss_comparison_only_not_calibration_or_closed_loop"}
    import transformers
    import inspect
    from openpi.models_pytorch import gemma_pytorch, preprocessing_pytorch
    source_paths = [Path(__file__), Path(inspect.getfile(precision_snapshot)), Path(inspect.getfile(PI0Pytorch)),
                    Path(inspect.getfile(gemma_pytorch)), Path(inspect.getfile(preprocessing_pytorch))]
    report["source_sha256"] = {str(path.resolve()): file_hash(path) for path in source_paths}
    report["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__}
    report.update(audit_schema(args.checkpoint, config))
    args.output.mkdir(parents=True, exist_ok=False)
    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    if report["schema_match"] != "PASS":
        raise RuntimeError(f"Schema mismatch; see {args.output / 'report.json'}")
    if args.mode == "forward":
        try:
            if args.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available in the target environment")
            torch.manual_seed(reference["seed"])
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            # Full digest only for the user-run forward, never the header-only inspection.
            report["checkpoint_sha256"] = file_hash(args.checkpoint / "model.safetensors")
            model, aliases = load_target(args.checkpoint, config, args.device)
            report["runtime_precision_after_constructor"] = precision_snapshot()
            report["runtime_precision_before_forward"] = configure_matmul_precision(args.float32_matmul_precision)
            report.update(strict_load="PASS", tied_aliases_filled=aliases,
                          model_weight_storage_allocated=True,
                          loaded_parameter_dtype_counts=dict(Counter(str(p.dtype) for p in model.parameters())))
            observation, actions, noise, time = observation_from_arrays(arrays, args.device)
            with torch.inference_mode():
                loss = model(observation, actions, noise=noise, time=time)
            actual = loss.float().cpu().numpy()
            comparison = compare_losses(actual, arrays["per_element_loss"], atol=args.atol, rtol=args.rtol)
            comparison["worst_index"] = [int(i) for i in comparison["worst_index"]]
            report.update(forward="PASS", input_identity="PASS", loss_shape=list(actual.shape),
                          loss_7d=float(actual[:, :, :7].mean()), loss_32d=float(actual.mean()),
                          reference_loss_7d=reference["loss_7d"], reference_loss_32d=reference["loss_32d"],
                          seed=reference["seed"], time=reference["time"], comparison=comparison,
                          peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else 0)
            np.savez(args.output / "comparison.npz", target_per_element_loss=actual,
                     reference_per_element_loss=arrays["per_element_loss"],
                     difference=actual - arrays["per_element_loss"])
        except Exception as error:
            report.update(forward="FAIL", error_type=type(error).__name__, error=str(error))
            save()
            raise
    save()
    print(json.dumps(report, indent=2))
    print("Saved:", args.output)
    if report.get("comparison", {}).get("status") == "MISMATCH":
        raise SystemExit("Loss comparison mismatch; diagnostics saved. Do not proceed to calibration or loosen tolerances automatically.")
    print("This does not validate gradients, raw-input adaptation, quantization, or LIBERO success.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""LeRobot diagnostic: compare signed velocity and three gradients with readiness output."""
import argparse
import json
import os
from pathlib import Path

for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[name] = "1"

import numpy as np
import torch
from compat import load_strict
from runtime_precision import configure_matmul_precision

ROOT = Path(__file__).resolve().parents[2]
SELECTED = {
    "blk.0.attn_q.weight": "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight",
    "blk.17.attn_k.weight": "paligemma_with_expert.paligemma.model.language_model.layers.17.self_attn.k_proj.weight",
    "blk.17.attn_q.weight": "paligemma_with_expert.paligemma.model.language_model.layers.17.self_attn.q_proj.weight",
    "v.blk.0.attn_q.weight": "paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "signed_velocity_and_four_gradients_only", "loss_scope": "mean32_for_comparison",
              "full_fisher_computed": False}
    try:
        with np.load(ROOT / "reproduction/results/pi05-real-forward-01/inputs_and_reference.npz") as data:
            arrays = {k: data[k] for k in data.files}
        torch.manual_seed(42)
        policy = load_strict(ROOT / "models/pi05_libero", "cuda")
        report["runtime_precision"] = configure_matmul_precision("highest")
        model = policy.model
        params = dict(model.named_parameters())
        def tensor(key):
            return torch.from_numpy(arrays[key]).cuda()
        cameras = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        captured = {}
        handle = model.action_out_proj.register_forward_hook(lambda m, inp, out: captured.update(velocity=out.detach().cpu().numpy()))
        loss = model([tensor("images." + k) for k in cameras], [tensor("image_masks." + k) for k in cameras],
                     tensor("observation.language.tokens"), tensor("observation.language.attention_mask").bool(),
                     tensor("actions"), noise=tensor("noise"), time=tensor("time"))
        handle.remove()
        unused_names = ["paligemma_with_expert.paligemma.model.language_model.layers.17." + suffix
                        for suffix in ("self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")]
        gradients_all = torch.autograd.grad(loss.mean(), [params[key] for key in [*SELECTED.values(), *unused_names]], allow_unused=True)
        gradients = gradients_all[:len(SELECTED)]
        if any(grad is not None for grad in gradients_all[len(SELECTED):]):
            raise ValueError("Unexpected final prefix gradient topology")
        report["final_prefix_unused_confirmed"] = unused_names
        captured["loss"] = loss.detach().cpu().numpy()
        for name, gradient in zip(SELECTED, gradients):
            captured[name] = gradient.detach().float().cpu().numpy()
        comparisons = {}
        with np.load(args.candidate / "selected_gradients.npz") as candidate:
            for name, expected in captured.items():
                actual = candidate[name]
                if actual.shape != expected.shape or not np.isfinite(expected).all() or not np.isfinite(actual).all():
                    raise ValueError(f"Invalid diagnostic array: {name}")
                delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
                comparisons[name] = {"shape": list(expected.shape), "exact_match": bool(np.array_equal(actual, expected)),
                                     "max_abs_diff": float(delta.max()), "mean_abs_diff": float(delta.mean())}
        report["comparisons"] = comparisons
        report["status"] = "EXACT_MATCH" if all(v["exact_match"] for v in comparisons.values()) else "MISMATCH"
        np.savez(args.output / "selected_gradients.npz", **captured)
        report["peak_gpu_gib"] = torch.cuda.max_memory_allocated()/2**30
    except Exception as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "EXACT_MATCH":
        raise SystemExit("Read diagnostic differences before proceeding")


if __name__ == "__main__":
    main()

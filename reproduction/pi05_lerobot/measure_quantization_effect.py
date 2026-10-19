#!/usr/bin/env python3
"""Measure lossy quantization against a paired, same-backend C++ baseline.

Finite execution and artifact checks are not task-quality acceptance tests.
The old strict cross-backend comparison remains in each probe directory.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def metrics(actual, reference):
    if actual.shape != reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("Invalid paired outputs")
    actual, reference = actual.astype(np.float64), reference.astype(np.float64)
    delta = actual-reference
    return {"max_abs_diff": float(np.max(np.abs(delta))), "mean_abs_diff": float(np.mean(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(delta**2))),
            "relative_l2": float(np.linalg.norm(delta)/max(np.linalg.norm(reference), 1e-30))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--quantized", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    b, q = (json.loads((p / "run.json").read_text()) for p in (args.baseline, args.quantized))
    if b["execution_returncode"] != 0 or q["execution_returncode"] != 0:
        raise ValueError("Both C++ executions must complete")
    if b["source_sha256"] != q["source_sha256"] or b["environment"] != q["environment"] or b["command"][3] != q["command"][3]:
        raise ValueError("Paired runtime, settings, or input path differs")
    if b["environment"]["ACTQUANT_PROBE_ODE_STEPS"] != "10":
        raise ValueError("Expected paired ten-step inference")

    def read(run, name, shape):
        return np.fromfile(run / "outputs" / (name+".f32"), dtype=np.float32).reshape(shape)

    baseline = read(args.baseline, "sampled_actions", (50,32))
    quantized = read(args.quantized, "sampled_actions", (50,32))
    trajectory_b = read(args.baseline, "ode_trajectory", (10,50,32))
    trajectory_q = read(args.quantized, "ode_trajectory", (10,50,32))
    stats = json.loads((args.model_dir / "norm_stats.json").read_text())
    mean, std = (np.asarray(stats[k], dtype=np.float64) for k in ("action_mean", "action_std"))
    if mean.shape != (7,) or std.shape != (7,):
        raise ValueError("Expected LIBERO action statistics")
    # Matches the pinned checkpoint processor's MEAN_STD inverse convention.
    denorm_b = baseline[:,:7]*(std+1e-8)+mean
    denorm_q = quantized[:,:7]*(std+1e-8)+mean
    velocity_b = read(args.baseline, "velocity_cpp_prefix", (50,32))
    velocity_q = read(args.quantized, "velocity_cpp_prefix", (50,32))
    report = {"status": "PAIRED_EXECUTION_PASS", "quality_acceptance": "NOT_EVALUATED",
              "scope": "one real observation; fixed seed-noise; paired C++ ten-step inference, no environment execution",
              "same_runtime_source": b["source_sha256"], "same_environment": b["environment"],
              "fixed_velocity_mean32": metrics(velocity_q, velocity_b),
              "sampled_actions_normalized_32d": metrics(quantized, baseline),
              "sampled_actions_normalized_7d": metrics(quantized[:,:7], baseline[:,:7]),
              "sampled_actions_denormalized_7d": metrics(denorm_q, denorm_b),
              "ode_step_normalized_7d": [metrics(trajectory_q[i,:,:7], trajectory_b[i,:,:7]) for i in range(10)],
              "baseline_first_action": denorm_b[0].tolist(), "quantized_first_action": denorm_q[0].tolist(),
              "previous_cross_backend_strict_status": "NUMERICAL_MISMATCH retained; no tolerance changes",
              "baseline_probe": str(args.baseline.resolve()), "quantized_probe": str(args.quantized.resolve())}
    saved = args.output.with_suffix(".npz")
    if saved.exists():
        raise FileExistsError(saved)
    np.savez(saved, baseline_actions=denorm_b, quantized_actions=denorm_q,
             baseline_normalized=baseline, quantized_normalized=quantized)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "ode_step_normalized_7d"}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare signed C++ intermediates; preserve strict diagnostic tolerances."""
import argparse
import json
from pathlib import Path

import numpy as np


def metrics(actual, expected):
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Invalid/non-finite comparison tensors")
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    delta = a-b
    close = np.abs(delta) <= 1e-6 + 1e-5*np.abs(b)
    return {"elements": a.size, "max_abs_diff": float(np.max(np.abs(delta))),
            "mean_abs_diff": float(np.mean(np.abs(delta))), "rmse": float(np.sqrt(np.mean(delta**2))),
            "relative_l2": float(np.linalg.norm(delta)/max(np.linalg.norm(b), 1e-30)),
            "cosine": float(np.dot(a, b)/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-30)),
            "elements_outside_tolerance": int(np.count_nonzero(~close)), "strict_allclose": bool(close.all())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    args = parser.parse_args()
    output = args.actual / "comparison.json"
    if output.exists():
        raise FileExistsError(output)

    def values(directory, name):
        return np.fromfile(directory / (name+".f32"), dtype=np.float32)

    pairs = {"image0": ("image_embedding0", "image_embedding_reference0"),
             "image1": ("image_embedding1", "image_embedding_reference1"),
             "text": ("text_embedding", "text_embedding_reference"),
             "time_embedding": ("timestep_embedding_cpp", "timestep_embedding"),
             "expert_same_prefix": ("velocity_reference_prefix", "velocity_reference"),
             "full_cpp_prefix": ("velocity_cpp_prefix", "velocity_reference"),
             "cached_cpp_prefix": ("velocity_cached_cpp_prefix", "velocity_reference")}
    report = {"scope": "export_precision_FP32_compute_reference_not_original_BF16_forward",
              "atol": 1e-6, "rtol": 1e-5,
              "comparisons": {label: metrics(values(args.actual, actual), values(args.reference, expected))
                              for label, (actual, expected) in pairs.items()}}
    report["cached_vs_joint"] = metrics(values(args.actual, "velocity_cached_cpp_prefix"), values(args.actual, "velocity_cpp_prefix"))
    report["raw_prompt_token_ids_match"] = bool(np.array_equal(
        np.fromfile(args.actual / "raw_prompt_tokens.i32", dtype=np.int32),
        np.fromfile(args.reference / "tokens.i32", dtype=np.int32)))
    report["numerical_status"] = "PASS" if all(v["strict_allclose"] for v in report["comparisons"].values()) else "MISMATCH"
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    if report["numerical_status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

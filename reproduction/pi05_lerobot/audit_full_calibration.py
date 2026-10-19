#!/usr/bin/env python3
"""Audit the full 60-episode/all-frame calibration dataset and finished Fisher export."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from calibration_adapter import load_raw, official_modules

EXPECTED = {"spatial": 10, "object": 10, "goal": 10, "long": 30}


def audit_data(directory):
    data = load_raw(directory)
    meta = data["meta"]
    if {s["suite"]: s["num_episodes"] for s in meta["suites"]} != EXPECTED:
        raise ValueError("Expected 10/10/10/30 episodes")
    offset = 0
    suites = []
    for suite in meta["suites"]:
        counts = Counter(ep["task"] for ep in suite["episodes"])
        expected_per_task = 3 if suite["suite"] == "long" else 1
        if len(suite["episodes"]) != EXPECTED[suite["suite"]] or len(counts) != 10 or set(counts.values()) != {expected_per_task}:
            raise ValueError(f"Incomplete task-stratified episodes: {suite['suite']}")
        frames = sum(ep["num_frames"] for ep in suite["episodes"])
        if frames != suite["num_frames"]:
            raise ValueError("Episode/frame count differs")
        for episode in suite["episodes"]:
            n = episode["num_frames"]
            if n < 1 or data["prompts"][offset:offset+n] != [episode["task"]]*n:
                raise ValueError("Episode task alignment differs")
            # Every frame is an origin. Verify all future windows and repeat-last tail.
            first_actions = data["gt_actions"][offset:offset+n, 0]
            for t in range(n):
                expected = first_actions[np.minimum(np.arange(t, t+50), n-1)]
                if not np.array_equal(data["gt_actions"][offset+t], expected):
                    raise ValueError(f"Action chunk/tail mismatch at frame {offset+t}")
            offset += n
        suites.append({"suite": suite["suite"], "episodes": len(suite["episodes"]), "tasks": len(counts), "frames": frames})
    if offset != meta["total_frames"]:
        raise ValueError("Dataset is missing frames")
    for key in ("state", "gt_actions"):
        if not np.isfinite(data[key]).all():
            raise ValueError(f"Non-finite {key}")
    pixels = data["pixel_values"]
    for start in range(0, len(pixels), 64):
        batch = pixels[start:start+64]
        if np.any(batch[:, 2]) or np.any(np.max(batch[:, :2], axis=(2, 3, 4)) == 0):
            raise ValueError("Missing real camera or nonzero placeholder")
    return {"status": "PASS", "scope": "60_stratified_episodes_all_frames", "frames": offset,
            "episodes": 60, "suites": suites, "all_action_windows_and_tails": "PASS", "finite_state_actions": "PASS",
            "real_cameras_and_zero_placeholder": "PASS", "image_shape": list(pixels.shape),
            "pixel_file_logical_bytes": (directory / "pixel_values.npy").stat().st_size,
            "pixel_file_allocated_bytes": (directory / "pixel_values.npy").stat().st_blocks*512}


def audit_fisher(directory, calibration):
    from gguf import GGUFReader, GGMLQuantizationType
    from safetensors import safe_open
    fisher, _ = official_modules()
    run = json.loads((directory / "run.json").read_text())
    total = calibration["frames"]
    if run.get("status") != "PASS" or run["num_samples"] != total or run.get("completed_samples", run["num_samples"]) != total:
        raise ValueError("Fisher did not finish all calibration frames")
    if not np.array_equal(np.load(directory / "indices.npy"), np.arange(total)):
        raise ValueError("Fisher indices do not cover all frames")
    losses = np.load(directory / "losses.npy")
    if losses.shape != (total,) or not np.isfinite(losses).all():
        raise ValueError("Invalid full Fisher losses")
    shapes = {}
    with safe_open(str(Path(run["checkpoint"]) / "model.safetensors"), framework="pt", device="cpu") as model:
        for name in model.keys():
            key = name.removeprefix("model.")
            mapped = fisher.hf_to_gguf_llm(key) or fisher.hf_to_gguf_vision(key)
            if mapped:
                shape = list(model.get_slice(name).get_shape())
                if mapped.startswith("v.") and mapped.endswith("ffn_up.weight"):
                    shape[0] += 48
                    shape[1] += 128
                elif mapped.startswith("v.") and mapped.endswith("ffn_down.weight"):
                    shape[1] += 48
                shapes[mapped] = shape
    checked = []
    for group, expected_count in (("llm", 126), ("vision", 162)):
        reader = GGUFReader(directory / f"fisher_{group}.gguf")
        if reader.get_field("fisher.num_samples").contents() != total or reader.get_field("imatrix.chunk_count").contents() != total:
            raise ValueError("Fisher metadata sample count mismatch")
        tensors = {t.name: t for t in reader.tensors}
        names = {name for name in shapes if name.startswith("v.") == (group == "vision")}
        if len(names) != expected_count or set(tensors) != {name+suffix for name in names for suffix in (".in_sum2", ".counts")}:
            raise ValueError("Fisher target set mismatch")
        for name in sorted(names):
            tensor = tensors[name+".in_sum2"]
            if tensor.tensor_type != GGMLQuantizationType.F32 or list(tensor.data.shape) != shapes[name]:
                raise ValueError(f"Fisher dtype/shape mismatch: {name}")
            values = tensor.data.reshape(-1)
            for start in range(0, values.size, 4_194_304):
                chunk = values[start:start+4_194_304]
                if not np.isfinite(chunk).all() or np.any(chunk <= 0):
                    raise ValueError(f"Non-positive/non-finite exported Fisher: {name}")
            if not np.array_equal(tensors[name+".counts"].data, np.ones((1, 1), np.float32)):
                raise ValueError("Fisher counts differ from mean-per-sample convention")
            checked.append(name)
    return {"status": "PASS", "samples": total, "targets": len(checked), "llm": 126, "vision": 162,
            "loss_mean": float(losses.mean()), "names_shapes_values_counts": "PASS"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calib-dir", type=Path, required=True)
    parser.add_argument("--fisher-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {"data": audit_data(args.calib_dir)}
    if args.fisher_dir:
        report["fisher"] = audit_fisher(args.fisher_dir, report["data"])
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

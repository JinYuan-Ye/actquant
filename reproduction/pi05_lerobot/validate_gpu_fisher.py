#!/usr/bin/env python3
"""Compare original and GPU Fisher on the real model with identical RNG and inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from calibration_adapter import ROOT, InputAdapter, load_raw, load_calibration_model, official_modules
import fisher_gpu_accumulation as fast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(8)
    fisher, _ = official_modules()
    adapter = InputAdapter()
    calib = load_raw(ROOT / "reproduction/data/pi05-full-60ep-01")
    model, _, _ = load_calibration_model(ROOT / "models/pi05_libero")
    targets = fisher.get_target_params(model)
    fisher.freeze_non_target(model, targets)
    original_accum = fisher.init_accumulators(targets)
    gpu_accum = fast.initialize(targets)
    report = {"status": "RUNNING", "matrices": len(targets), "samples": [],
              "gpu_execution_sha256": hashlib.sha256(Path(fast.__file__).read_bytes()).hexdigest()}
    for index in (0, 1216):
        obs, actions = adapter.build(calib, np.array([index]), "cuda")
        state = (np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
        torch.cuda.synchronize()
        start = time.monotonic()
        original_loss = fisher.fisher_step(model, targets, original_accum, obs, actions)
        original_seconds = time.monotonic()-start
        np.random.set_state(state[0])
        torch.set_rng_state(state[1])
        torch.cuda.set_rng_state_all(state[2])
        torch.cuda.synchronize()
        start = time.monotonic()
        gpu_loss = fast.step(model, targets, gpu_accum, obs, actions)
        gpu_seconds = time.monotonic()-start
        equal = all(torch.equal(original_accum[name], gpu_accum[name].cpu()) for name in original_accum)
        report["samples"].append({"index": index, "loss_original": original_loss, "loss_gpu": gpu_loss,
                                  "loss_exact": original_loss == gpu_loss, "all288_accumulators_exact": equal,
                                  "original_step_seconds": original_seconds, "gpu_step_seconds": gpu_seconds})
        print(report["samples"][-1], flush=True)
        if not equal or original_loss != gpu_loss:
            report["status"] = "MISMATCH"
            args.output.write_text(json.dumps(report, indent=2)+"\n")
            raise SystemExit(2)
    # Exercise exactly the same snapshot serialization/map_location and pop-to-NumPy export path.
    import tempfile
    small = fast.GPUAccumulatorMap({"test": gpu_accum[next(iter(gpu_accum))][:2, :3].clone()})
    with tempfile.TemporaryDirectory() as temporary:
        snapshot = Path(temporary) / "state.pt"
        torch.save({"accumulators": small}, snapshot)
        restored = torch.load(snapshot, map_location="cpu", weights_only=False)["accumulators"]
        if not np.array_equal(restored.pop("test").numpy(), dict.pop(small, "test").cpu().numpy()):
            raise ValueError("Snapshot/export conversion differs")
    report.update(status="PASS", snapshot_and_export="PASS", peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

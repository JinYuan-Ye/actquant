#!/usr/bin/env python3
"""Resume the audited runner with equivalent GPU accumulators and provenance."""
import hashlib
import json
from pathlib import Path
import sys

import torch
import run_calibration as runner
from fisher_gpu_accumulation import GPUAccumulatorMap, initialize, step


def main():
    # Keep run_calibration.py byte-identical so its exact snapshot/RNG contract remains valid.
    if "--stage" not in sys.argv or sys.argv[sys.argv.index("--stage")+1] != "fisher":
        raise ValueError("GPU execution wrapper supports Fisher only")
    validation = runner.ROOT / "reproduction/results/pi05-full-01/gpu-fisher-validation.json"
    checked = json.loads(validation.read_text())
    module_path = Path(__file__).with_name("fisher_gpu_accumulation.py")
    module_sha = hashlib.sha256(module_path.read_bytes()).hexdigest()
    if checked.get("status") != "PASS" or checked.get("gpu_execution_sha256") != module_sha:
        raise ValueError("GPU Fisher execution must pass the real-model numerical validation first")
    torch.set_num_threads(8)
    original_modules = runner.official_modules
    original_load = torch.load
    original_precision = runner.precision_snapshot

    def modules():
        fisher, hsic = original_modules()
        fisher.init_accumulators = initialize
        fisher.fisher_step = step
        return fisher, hsic

    def load(*args, **kwargs):
        state = original_load(*args, **kwargs)
        if isinstance(state, dict) and "accumulators" in state:
            state["accumulators"] = GPUAccumulatorMap(state["accumulators"])
        return state

    def precision():
        return {**original_precision(), "fisher_execution": "equivalent_gpu_accumulators",
                "gpu_execution_sha256": module_sha, "validation": str(validation),
                "cpu_threads": torch.get_num_threads(), "per_sample_empty_cache": False}

    runner.official_modules = modules
    runner.precision_snapshot = precision
    torch.load = load
    torch.cuda.empty_cache = lambda: None
    runner.main()


if __name__ == "__main__":
    main()

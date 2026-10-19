#!/usr/bin/env python3
"""Measure the original Fisher accumulation overhead at real target shapes."""
import argparse
import json
from pathlib import Path
import time

import torch
from safetensors import safe_open
from calibration_adapter import ROOT, official_modules


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    fisher, _ = official_modules()
    shapes = {}
    with safe_open(str(ROOT / "models/pi05_libero/model.safetensors"), framework="pt", device="cpu") as source:
        for name in source.keys():
            key = name.removeprefix("model.")
            mapped = fisher.hf_to_gguf_llm(key) or fisher.hf_to_gguf_vision(key)
            if mapped:
                shapes[mapped] = source.get_slice(name).get_shape()
    default_threads = torch.get_num_threads()
    torch.set_num_threads(4)
    torch.manual_seed(1337)
    gradients = {name: torch.randn(shape, dtype=torch.bfloat16, device="cuda")*.01 for name, shape in shapes.items()}
    accumulators = {name: torch.full(shape, .125, dtype=torch.float32) for name, shape in shapes.items()}
    report = {"scope": "accumulation_only_at_actual288_target_shapes; no model forward", "default_cpu_threads": default_threads,
              "matrices": len(shapes), "elements": sum(g.numel() for g in gradients.values()), "cases": {}}
    for threads in (default_threads, 4):
        torch.set_num_threads(4)
        for value in accumulators.values():
            value.fill_(.125)
        torch.set_num_threads(threads)
        torch.cuda.synchronize()
        started = time.monotonic()
        transfer = addition = 0.
        for name, grad in gradients.items():
            now = time.monotonic()
            squared_cpu = grad.detach().float().pow(2).cpu()
            transfer += time.monotonic()-now
            now = time.monotonic()
            accumulators[name] += squared_cpu
            addition += time.monotonic()-now
        report["cases"][f"cpu_accum_threads_{threads}"] = {"seconds": time.monotonic()-started,
                                                               "square_and_d2h_seconds": transfer, "cpu_add_seconds": addition}
        print(f"CPU threads {threads}: {report['cases'][f'cpu_accum_threads_{threads}']}", flush=True)
    gpu_accumulators = {name: torch.full(shape, .125, dtype=torch.float32, device="cuda") for name, shape in shapes.items()}
    torch.cuda.synchronize()
    started = time.monotonic()
    for name, grad in gradients.items():
        gpu_accumulators[name] += grad.detach().float().pow(2)
    torch.cuda.synchronize()
    report["cases"]["gpu_accum"] = {"seconds": time.monotonic()-started}
    equal = all(torch.equal(accumulators[name], gpu_accumulators[name].cpu()) for name in accumulators)
    report.update(bitwise_equal_all288=equal, status="PASS" if equal else "MISMATCH",
                  peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)
    if not equal:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

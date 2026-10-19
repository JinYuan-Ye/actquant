#!/usr/bin/env python3
"""Audit all LLM quantization assignments and merge a complete pi05 artifact.

Consumes a finished independent llama-quantize run. Keeps original embedding,
norms, vision and expert. This does not certify policy quality or closed loop.
"""
import argparse
from collections import Counter
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType

from merge_gguf_compat import merge

ROOT = Path(__file__).resolve().parents[2]


def exact_bytes(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    av, bv = a.view(np.uint8).reshape(-1), b.view(np.uint8).reshape(-1)
    return all(np.array_equal(av[i:i+4_194_304], bv[i:i+4_194_304]) for i in range(0,av.size,4_194_304))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, default=ROOT / "reproduction/results/pi05-allocation-demo-01/allocation.json")
    parser.add_argument("--base-dir", type=Path, default=ROOT / "reproduction/results/pi05-quant-bridge-01/unified")
    parser.add_argument("--imatrix", type=Path, default=ROOT / "reproduction/results/pi05-fisher-demo-01/fisher_llm.gguf")
    args = parser.parse_args()
    output = args.run_dir / "artifact-audit.json"
    if output.exists():
        raise FileExistsError(output)
    run = json.loads((args.run_dir / "run.json").read_text())
    if run.get("status") != "QUANTIZATION_FINISHED" or run.get("returncode") != 0:
        raise ValueError("Quantization must finish successfully before audit/merge")
    start = time.monotonic()
    assigned = json.loads(args.allocation.read_text())
    assignments = assigned["assignments"]
    expected_names = {f"blk.{i}.{kind}.weight" for i in range(18) for kind in
                      ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")}
    if set(assignments) != expected_names:
        raise ValueError("Expected exact 126 target assignments")
    log = (args.run_dir / "quantize.log").read_text()
    if log.count("(per-weight)") != 126:
        raise ValueError("Not all 126 targets logged per-weight Fisher")
    quantized = GGUFReader(args.run_dir / "pali_llm_mixed.gguf")
    weights = {tensor.name: tensor for tensor in quantized.tensors}
    importance_reader = GGUFReader(args.imatrix)
    importance = {tensor.name: tensor for tensor in importance_reader.tensors}
    calibration_samples = int(importance_reader.get_field("fisher.num_samples").contents())
    if calibration_samples < 1 or calibration_samples != int(importance_reader.get_field("imatrix.chunk_count").contents()):
        raise ValueError("Invalid Fisher calibration sample metadata")
    fisher_report_path = args.imatrix.parent / "run.json"
    fisher_report = json.loads(fisher_report_path.read_text())
    if fisher_report.get("status") != "PASS" or fisher_report["num_samples"] != calibration_samples:
        raise ValueError("Fisher run provenance differs from important-file metadata")
    data_metadata = fisher_report["data_metadata"]
    suite_counts = {s["suite"]: s["num_episodes"] for s in data_metadata.get("suites", [])}
    full_calibration = suite_counts == {"spatial": 10, "object": 10, "goal": 10, "long": 30} and calibration_samples == data_metadata["total_frames"]
    calibration_scope = "all_frames_of_stratified_60_episodes" if full_calibration else "limited_calibration_not_full_60_episodes"
    params = payload = 0
    for name, kind in assignments.items():
        tensor = weights[name]
        if tensor.tensor_type.name != kind or list(tensor.shape) != list(importance[name+".in_sum2"].shape):
            raise ValueError(f"Quantization type/shape mismatch: {name}")
        params += tensor.n_elements
        payload += tensor.n_bytes
    actual_bpw = payload*8/params
    if actual_bpw > assigned["target_bpw"] or abs(actual_bpw-assigned["achieved_bpw"]) > 1e-12:
        raise ValueError("Actual encoded target bpw differs from allocation")
    if weights["token_embd.weight"].tensor_type != GGMLQuantizationType.F16 or weights["output.weight"].tensor_type != GGMLQuantizationType.F16:
        raise ValueError("Standalone embeddings/output must stay F16 for this run")
    model_dir = args.run_dir / "model"
    model_dir.mkdir(exist_ok=False)
    with (args.run_dir / "merge.log").open("x") as log_file, contextlib.redirect_stdout(log_file):
        merge(args.base_dir / "pi05.gguf", args.run_dir / "pali_llm_mixed.gguf", model_dir / "pi05.gguf", f"HSIC_Fisher_mixed_{calibration_samples}samples")
    base, merged = (GGUFReader(path) for path in (args.base_dir / "pi05.gguf", model_dir / "pi05.gguf"))
    before, after = ({tensor.name: tensor for tensor in reader.tensors} for reader in (base, merged))
    if set(before) != set(after):
        raise ValueError("Merged tensor set differs from complete base")
    spec = importlib.util.spec_from_file_location("quant_finish_official_merge", ROOT / "src/ActQuant/tools/pi0.5/merge_pi05_llm.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    routed = {official.llm_name_to_pali(name): name for name in assignments}
    unchanged = 0
    for name, tensor in after.items():
        expected = weights[routed[name]] if name in routed else before[name]
        if tensor.tensor_type != expected.tensor_type or not np.array_equal(tensor.shape, expected.shape) or not exact_bytes(tensor.data, expected.data):
            raise ValueError(f"Merged tensor differs from selected source: {name}")
        unchanged += name not in routed
    checked_metadata = 0
    for key, field in base.fields.items():
        if key.startswith("GGUF.") or key == "pi05.quant_llm":
            continue
        if field.contents() != merged.get_field(key).contents():
            raise ValueError(f"Metadata changed unexpectedly: {key}")
        checked_metadata += 1
    for filename in ("tokenizer.model", "norm_stats.json"):
        shutil.copy2(args.base_dir / filename, model_dir / filename)
    base_bytes = (args.base_dir / "pi05.gguf").stat().st_size
    quant_bytes = (model_dir / "pi05.gguf").stat().st_size
    report = {"status": "ARTIFACT_PASS", "scope": "complete_pi05_container_with_all126_LLM_weights_quantized",
              "calibration_samples": calibration_samples, "calibration_scope": calibration_scope,
              "full_calibration": full_calibration, "calibration_suite_episodes": suite_counts,
              "quality_status": "requires_same_backend_forward_and_closed_loop_evaluation",
              "per_weight_fisher_log_count": 126, "quantized_llm_matrices": len(assignments),
              "unchanged_tensors": unchanged, "total_tensors": len(after), "metadata_fields_preserved": checked_metadata,
              "actual_block_bpw": actual_bpw, "encoded_block_bytes": payload, "format_counts": dict(Counter(assignments.values())),
              "base_gguf_bytes": base_bytes, "quantized_gguf_bytes": quant_bytes, "size_ratio_base_over_quantized": base_bytes/quant_bytes,
              "size_reduction_fraction": 1-quant_bytes/base_bytes, "model_dir": str(model_dir.resolve()),
              "unchanged_components": ["vision", "embedding", "action_expert", "projector", "norms", "normalization_stats"],
              "cross_backend_strict_result": "prior numerical mismatch retained, not relabeled PASS",
              "seconds": time.monotonic()-start}
    output.write_text(json.dumps(report, indent=2)+"\n")
    (model_dir / "quantization_manifest.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

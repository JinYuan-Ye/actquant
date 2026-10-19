#!/usr/bin/env python3
"""Audit official exports and quantize a bounded set of real checkpoint tensors.

Requires existing unified/ and llm/ exports. Creates selected/ exclusively.
The resulting partial files are quantizer fixtures, not inference models.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
import torch
from safetensors import safe_open
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as QT
from gguf.quants import dequantize

ROOT = Path(__file__).resolve().parents[2]


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def copy_metadata(reader, writer):
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        writer.add_key_value(key, field.contents(), field.types[0],
                             field.types[1] if len(field.types) > 1 else None)


def write_subset(reader, names, path, architecture, transforms=None):
    writer = GGUFWriter(str(path), architecture)
    copy_metadata(reader, writer)
    tensors = {t.name: t for t in reader.tensors}
    for name in sorted(names):
        t = tensors[name]
        data = transforms[name] if transforms and name in transforms else t.data
        writer.add_tensor(name, np.ascontiguousarray(data), raw_dtype=t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/pi05_libero")
    parser.add_argument("--fisher-dir", type=Path, default=ROOT / "reproduction/results/pi05-fisher-demo-01")
    parser.add_argument("--allocation", type=Path, default=ROOT / "reproduction/results/pi05-allocation-demo-01/allocation.json")
    parser.add_argument("--work-name", default="selected", help="New subdirectory for this attempt")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.cpu_threads < 1 or Path(args.work_name).name != args.work_name:
        parser.error("Need positive CPU threads and a simple work directory name")
    torch.set_num_threads(args.cpu_threads)
    selected = args.export_dir / args.work_name
    selected.mkdir(exist_ok=False)
    if (args.export_dir / "bridge-audit.json").exists():
        raise FileExistsError(args.export_dir / "bridge-audit.json")
    report = {"scope": "all_target_export_values_and_bounded_real_tensor_quantization_not_inference",
              "commands": [], "warnings": []}
    start = time.monotonic()

    def save():
        (args.export_dir / "bridge-audit.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(label, command):
        with (selected / (label + ".log")).open("x") as log:
            proc = subprocess.run(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT)
        report["commands"].append({"label": label, "command": list(map(str, command)), "returncode": proc.returncode})
        save()
        if proc.returncode:
            raise RuntimeError(f"{label} failed; see {selected / (label + '.log')}")
        return (selected / (label + ".log")).read_text()

    try:
        fisher = module("bridge_fisher", "src/ActQuant/tools/fisher-diag/compute_fisher_pi05.py")
        merge = module("bridge_merge", "src/ActQuant/tools/pi0.5/merge_pi05_llm.py")
        base = GGUFReader(args.export_dir / "unified/pi05.gguf")
        llm = GGUFReader(args.export_dir / "llm/pali_llm_bf16.gguf")
        bt, lt = ({t.name: t for t in r.tensors} for r in (base, llm))
        mapping = {}
        with safe_open(args.checkpoint / "model.safetensors", framework="pt", device="cpu") as source:
            for key in source.keys():
                name = key.removeprefix("model.")
                target = fisher.hf_to_gguf_llm(name) or fisher.hf_to_gguf_vision(name)
                if target:
                    mapping[target] = key
            if len(mapping) != 288:
                raise ValueError(f"Expected 288 targets, got {len(mapping)}")
            for name, key in mapping.items():
                src = source.get_tensor(key)
                unified_name = name if name.startswith("v.") else merge.llm_name_to_pali(name)
                targets = [bt[unified_name]] + ([] if name.startswith("v.") else [lt[name]])
                for tensor in targets:
                    expected_shape = list(src.shape)
                    if name.startswith("v.") and name.endswith("ffn_up.weight"):
                        expected_shape[0] += 48
                    elif name.startswith("v.") and name.endswith("ffn_down.weight"):
                        expected_shape[1] += 48
                    if tensor.tensor_type != QT.F16 or list(tensor.data.shape) != expected_shape:
                        raise ValueError(f"Unexpected exported shape/type: {tensor.name}")
                    for offset in range(0, src.shape[0], 64):
                        expected = src[offset:offset+64].float().half().numpy()
                        actual = tensor.data[offset:offset+len(expected), :src.shape[1]]
                        if not np.array_equal(actual.view(np.uint16), expected.view(np.uint16)):
                            raise ValueError(f"FP16 export bits differ: {tensor.name}, row {offset}")
                    if name.startswith("v.") and name.endswith("ffn_up.weight"):
                        assert np.all(tensor.data[-48:, :] == 0)
                    elif name.startswith("v.") and name.endswith("ffn_down.weight"):
                        assert np.all(tensor.data[:, -48:] == 0)
            norms_checked = 0
            for i in range(18):
                for hf, gg in [("input_layernorm", "attn_norm"), ("post_attention_layernorm", "ffn_norm")]:
                    src = source.get_tensor(f"model.paligemma_with_expert.paligemma.model.language_model.layers.{i}.{hf}.weight").float().numpy()
                    np.testing.assert_array_equal(bt[f"pali.blk.{i}.{gg}.weight"].data, src)
                    np.testing.assert_array_equal(lt[f"blk.{i}.{gg}.weight"].data, 1.0 + src)
                    assert merge.llm_name_to_pali(f"blk.{i}.{gg}.weight") is None
                    norms_checked += 1
            embed = source.get_tensor("model.paligemma_with_expert.paligemma.lm_head.weight")
            for offset in range(0, embed.shape[0], 512):
                expected = embed[offset:offset+512].float().half().numpy()
                for name, table in [("embed.weight", bt), ("token_embd.weight", lt), ("output.weight", lt)]:
                    if not np.array_equal(table[name].data[offset:offset+len(expected)].view(np.uint16), expected.view(np.uint16)):
                        raise ValueError(f"Embedding export bits differ: {name}, row {offset}")
        report["export_values"] = {"status": "PASS", "target_matrices": len(mapping), "paired_layer_norms": norms_checked,
                                   "tied_embedding_copies_match_source": True, "matrix_storage": "F16"}
        assert base.get_field("pi05_action.action_horizon").contents() == 50
        assert base.get_field("pi05_action.actual_action_dim").contents() == 7
        assert base.get_field("pi05_action.actual_state_dim").contents() == 8
        for tensor in base.tensors:
            if tensor.data.dtype == np.float16:
                assert np.all((tensor.data.view(np.uint16) & 0x7c00) != 0x7c00), tensor.name
            else:
                assert np.isfinite(tensor.data).all(), tensor.name
        # Probe the official merge metadata copier without writing another full model.
        probe = GGUFWriter(None, "pi05")
        merge.copy_kv_metadata(base, probe, None)
        bad_arrays = []
        for key in ("clip.vision.image_mean", "clip.vision.image_std"):
            original = base.get_field(key).contents()
            copied = list(probe.kv_data[0][key].value)
            if original != copied:
                bad_arrays.append({"key": key, "original": original, "official_copy": copied})
        report["official_merge_array_copy_probe"] = bad_arrays
        if bad_arrays:
            report["warnings"].append("Official merge metadata copier truncates numeric arrays; use contents-based copying in external bridge.")
        alloc = json.loads(args.allocation.read_text())["assignments"]
        # All 18 V matrices satisfy llama-quantize's attention-count consistency assertion.
        names = {f"blk.{i}.attn_v.weight" for i in range(18)} | {
            "blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.ffn_up.weight",
            "blk.17.attn_q.weight", "blk.13.attn_k.weight"}
        write_subset(llm, names, selected / "llm-input.gguf", "llama")
        fi = GGUFReader(args.fisher_dir / "fisher_llm.gguf")
        im_names = {name+suffix for name in names for suffix in (".in_sum2", ".counts")}
        write_subset(fi, im_names, selected / "llm-imatrix.gguf", "")
        command = [ROOT / "build/pi05/bin/llama-quantize", "--imatrix", selected / "llm-imatrix.gguf"]
        # Explicitly override every target, including base types. Do not use --pure:
        # this pinned C++ implementation handles overrides inside the !pure branch.
        for name in sorted(names):
            command.extend(["--tensor-type", "^" + re.escape(name) + "$=" + alloc[name]])
        log = run("llm-explicit", command + [selected / "llm-input.gguf", selected / "llm-quantized.gguf", "IQ2_XS", "4"])
        assert log.count("(per-weight)") == len(names)
        quant = GGUFReader(selected / "llm-quantized.gguf")
        assert {t.name for t in quant.tensors} == names
        reconstruction = {}
        for t in quant.tensors:
            assert t.tensor_type.name == alloc[t.name], (t.name, t.tensor_type.name, alloc[t.name])
            decoded = dequantize(t.data, t.tensor_type)
            assert np.isfinite(decoded).all() and decoded.shape == lt[t.name].data.shape
            reconstruction[t.name] = {"type": t.tensor_type.name, "shape": list(decoded.shape),
                                      "mse_vs_exported_f16": float(np.mean((decoded.astype(np.float64)-lt[t.name].data.astype(np.float64))**2))}
        report["llm_quantization"] = {"status": "PASS", "matrices": len(names), "per_weight_log_count": log.count("(per-weight)"), "reconstruction": reconstruction}
        # Test the official driver strategy on the same bounded fixture.
        legacy_command = [ROOT / "build/pi05/bin/llama-quantize", "--imatrix", selected / "llm-imatrix.gguf"]
        for name in sorted(names):
            if alloc[name] != "IQ2_XS":
                legacy_command.extend(["--tensor-type", "^" + re.escape(name) + "$=" + alloc[name]])
        run("llm-driver-strategy", legacy_command + [selected / "llm-input.gguf", selected / "llm-driver-strategy.gguf", "IQ2_XS", "4"])
        legacy = GGUFReader(selected / "llm-driver-strategy.gguf")
        differences = {t.name: {"assigned": alloc[t.name], "actual": t.tensor_type.name}
                       for t in legacy.tensors if alloc[t.name] != t.tensor_type.name}
        report["official_driver_format_probe"] = differences
        if differences:
            report["warnings"].append("Overriding upgraded tensors only leaves preset heuristics active for base assignments; explicit overrides required.")
        vnames = {"v.blk.0.attn_q.weight", "v.blk.0.ffn_up.weight", "v.blk.0.ffn_down.weight"}
        write_subset(base, vnames, selected / "vision-input.gguf", "pi05")
        vf = GGUFReader(args.fisher_dir / "fisher_vision.gguf")
        vt = {t.name: t for t in vf.tensors}
        transforms, alignment = {}, {}
        for name in vnames:
            shape = bt[name].data.shape
            data = vt[name+".in_sum2"].data
            if data.shape != shape:
                assert name.endswith("ffn_up.weight") and data.shape == (shape[0], shape[1]+128)
                transforms[name+".in_sum2"] = data[:, :shape[1]]
                alignment[name] = {"fisher_shape": list(data.shape), "input_shape": list(shape),
                                   "operation": "remove 128 padded-only columns before C++ tool re-pads"}
        write_subset(vf, {n+s for n in vnames for s in (".in_sum2", ".counts")},
                     selected / "vision-imatrix-aligned.gguf", "", transforms)
        vlog = run("vision-aligned", [ROOT / "build/pi05/bin/quantize-vision", selected / "vision-input.gguf",
                   selected / "vision-quantized.gguf", "Q4_K_M", "--imatrix", selected / "vision-imatrix-aligned.gguf"])
        assert vlog.count("[imatrix: per-weight]") == len(vnames)
        assert "ignoring" not in vlog and "[imatrix: missing]" not in vlog
        vr = GGUFReader(selected / "vision-quantized.gguf")
        for t in vr.tensors:
            assert t.tensor_type == QT.Q4_K
            assert np.isfinite(dequantize(t.data, t.tensor_type)).all()
        report["vision_quantization"] = {"status": "PASS", "matrices": len(vnames), "per_weight_log_count": len(vnames), "alignment": alignment}
        report.update(status="PASS", duration_seconds=time.monotonic()-start)
    except Exception as error:
        report.update(status="FAIL", error=repr(error), duration_seconds=time.monotonic()-start)
        save()
        raise
    save()
    print(json.dumps({k: v for k, v in report.items() if k not in ("commands", "llm_quantization")}, indent=2))


if __name__ == "__main__":
    main()

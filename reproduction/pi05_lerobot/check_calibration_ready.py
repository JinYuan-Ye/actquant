#!/usr/bin/env python3
"""Bounded readiness check: raw-input replay, hooks, and one backward. No calibration scores."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from calibration_adapter import (ROOT, InputAdapter, load_raw, load_calibration_model,
                                 official_modules, experiment_contract)
from openpi_compare import CAMERAS, file_hash, compare_losses
from prepare_real_probe import DEFAULT_DATA, SHARD, SHARD_SHA256, load_exporter, select_sample, sha256
from runtime_precision import precision_snapshot


def make_small_dataset(output):
    if output.exists():
        raise FileExistsError(output)
    shard = DEFAULT_DATA / SHARD
    if sha256(shard) != SHARD_SHA256:
        raise ValueError("Pinned RLDS shard hash mismatch")
    exporter, _ = load_exporter()
    episode = next(exporter.iter_episodes([shard]))
    frames = [0, 10, 20, 30]
    samples = [select_sample(episode, frame, 50, exporter, "native") for frame in frames]
    output.mkdir(parents=True, exist_ok=False)
    for field, index in (("pixel_values", 0), ("state", 1), ("gt_actions", 2)):
        np.save(output / (field + ".npy"), np.concatenate([s[index] for s in samples], axis=0))
    (output / "prompts.json").write_text(json.dumps([s[3] for s in samples]) + "\n")
    (output / "metadata.json").write_text(json.dumps({"total_frames": len(frames),
        "image_mode": "native", "camera_order": list(CAMERAS), "action_horizon": 50,
        "source_shard": SHARD, "shard_sha256": SHARD_SHA256, "episode": 0,
        "source_frames": frames, "action_tail_padding": False,
        "sample_kind": "readiness_only_not_paper_calibration_split"}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data-output", type=Path)
    group.add_argument("--calib-dir", type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.data_output is not None and args.data_output.exists()):
        raise FileExistsError("Use fresh readiness result and data directories")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"contract": experiment_contract(), "full_calibration_run": False}
    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    start = time.monotonic()
    try:
        if args.data_output is not None:
            make_small_dataset(args.data_output)
        calib_dir = args.calib_dir or args.data_output
        calib = load_raw(calib_dir)
        report["calib_dir"] = str(calib_dir)
        adapter = InputAdapter()
        with np.load(ROOT / "reproduction/results/pi05-real-forward-01/inputs_and_reference.npz") as data:
            reference = {key: data[key] for key in data.files}
        obs, actions = adapter.build(calib, np.array([0]), "cuda")
        actual_inputs = {"actions": actions, "observation.state": obs.state,
                         "observation.language.tokens": obs.tokenized_prompt,
                         "observation.language.attention_mask": obs.tokenized_prompt_mask}
        actual_inputs.update({"images." + k: v for k, v in obs.images.items()})
        actual_inputs.update({"image_masks." + k: v for k, v in obs.image_masks.items()})
        for key, value in actual_inputs.items():
            np.testing.assert_array_equal(value.cpu().numpy(), reference[key], err_msg=key)
        report["raw_input_matches_reference"] = "EXACT_MATCH"
        print("Raw input adaptation: EXACT_MATCH", flush=True)
        save()
        torch.manual_seed(42)
        model, config, aliases = load_calibration_model(ROOT / "models/pi05_libero")
        report.update(strict_load="PASS", tied_aliases_filled=aliases, runtime_precision=precision_snapshot())
        noise = torch.from_numpy(reference["noise"]).cuda()
        moment = torch.from_numpy(reference["time"]).cuda()
        signed = {}
        handle = model.action_out_proj.register_forward_hook(lambda module, inp, out: signed.update(velocity=out.detach().cpu().numpy()))
        with torch.no_grad():
            per_element = model(obs, actions, noise=noise, time=moment)
        handle.remove()
        comparison = compare_losses(per_element.cpu().numpy(), reference["per_element_loss"], atol=1e-6, rtol=1e-5)
        comparison["worst_index"] = [int(i) for i in comparison["worst_index"]]
        report["raw_forward_comparison"] = comparison
        if comparison["status"] != "EXACT_MATCH":
            raise RuntimeError("Raw-input forward no longer exactly matches saved reference")
        print("Raw-input model forward: EXACT_MATCH", flush=True)
        fisher, hsic = official_modules()
        targets = fisher.get_target_params(model)
        groups = {group: sum(t[3] == group for t in targets) for group in ("llm", "vision")}
        if groups != {"llm": 126, "vision": 162}:
            raise ValueError(f"Unexpected original target mapping: {groups}")
        report["target_matrices"] = groups
        # Exercise the original hooks on 4 real frames, but do not compute any HSIC scores.
        original_builder = hsic.build_observation_and_actions
        hsic.build_observation_and_actions = adapter.official_builder
        try:
            X, Y, inputs, outputs = hsic.collect_activations_pi05(model, calib, np.arange(4), None, None,
                torch.device("cuda"), 50, 32, 32, 200, 1, 0)
        finally:
            hsic.build_observation_and_actions = original_builder
        if X.shape != (4, 2048) or Y.shape != (4, 350) or len(inputs) != 126 or len(outputs) != 126:
            raise ValueError("Original activation collector returned incompatible shapes")
        if not all(np.isfinite(a).all() for a in [X, Y, *inputs.values(), *outputs.values()]):
            raise ValueError("Non-finite activation output")
        report["activation_collection"] = {"status": "PASS", "frames": 4, "X_shape": list(X.shape),
                                           "Y_shape": list(Y.shape), "matrices": len(outputs),
                                           "hsic_scores_computed": False}
        del X, Y, inputs, outputs, per_element
        print("Original HSIC hooks: PASS (126 matrices, four real frames)", flush=True)
        save()
        # Exactly the original mean32 loss and all original target gradients, without squaring/accumulating Fisher.
        fisher.freeze_non_target(model, targets)
        per_element = model(obs, actions, noise=noise, time=moment)
        grads = torch.autograd.grad(per_element.mean(), [t[2] for t in targets], allow_unused=True)
        selected_names = {"blk.0.attn_q.weight", "blk.17.attn_k.weight", "blk.17.attn_q.weight", "v.blk.0.attn_q.weight"}
        selected_arrays = {"velocity": signed["velocity"], "loss": per_element.detach().cpu().numpy()}
        missing, nonfinite, zero = [], [], []
        for (_, name, parameter, _), grad in zip(targets, grads):
            if grad is None:
                missing.append(name)
                continue
            if grad.shape != parameter.shape or not torch.isfinite(grad).all():
                nonfinite.append(name)
            if torch.count_nonzero(grad) == 0:
                zero.append(name)
            if name in selected_names:
                selected_arrays[name] = grad.detach().float().cpu().numpy()
        # Final prefix output is discarded: its O/MLP are unused, and prefix Q
        # receives zero gradient. Final prefix K/V still influence suffix actions.
        expected_missing = {"blk.17.attn_output.weight", "blk.17.ffn_gate.weight",
                            "blk.17.ffn_up.weight", "blk.17.ffn_down.weight"}
        expected_zero = {"blk.17.attn_q.weight"}
        valid = set(missing) == expected_missing and set(zero) == expected_zero and not nonfinite
        report["backward"] = {"status": "PASS" if valid else "FAIL",
            "target_count": len(targets), "missing": missing, "invalid": nonfinite, "all_zero": zero,
            "expected_unused_final_prefix": sorted(expected_missing), "expected_zero_final_prefix": sorted(expected_zero),
            "loss_scope": "mean over full32", "optimizer_steps": 0, "fisher_accumulators_created": False}
        if not valid:
            raise RuntimeError("Original target gradient connectivity check failed")
        np.savez(args.output / "selected_gradients.npz", **selected_arrays)
        report.update(status="PASS", peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                      duration_seconds=time.monotonic()-start,
                      adapter_sha256=file_hash(ROOT / "reproduction/pi05_lerobot/calibration_adapter.py"),
                      reference_sha256=file_hash(ROOT / "reproduction/results/pi05-real-forward-01/inputs_and_reference.npz"))
        print("Backward: PASS (288 targets; expected final-prefix unused/zero gradients recorded)", flush=True)
    except Exception as error:
        report.update(status="FAIL", error=repr(error), duration_seconds=time.monotonic()-start)
        save()
        raise
    save()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

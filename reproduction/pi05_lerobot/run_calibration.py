#!/usr/bin/env python3
"""Single-GPU glue around official ActQuant routines using audited LeRobot inputs.

This is an explicit experiment entry point. Read METHOD_WALKTHROUGH.md before use.
"""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from calibration_adapter import (ROOT, InputAdapter, load_raw, load_calibration_model,
                                 official_modules, experiment_contract)
from openpi_compare import file_hash
from runtime_precision import precision_snapshot


def pad_vision_for_export(name, array):
    # Same output-layout padding as official compute_fisher_pi05.main, not a new statistic.
    if name.endswith("ffn_down.weight"):
        return np.pad(array, ((0, 0), (0, 48)), mode="constant")
    if name.endswith("ffn_up.weight"):
        return np.pad(array, ((0, 48), (0, 128)), mode="constant")
    return array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("hsic", "fisher"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/pi05_libero")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "models/paligemma_tokenizer")
    parser.add_argument("--calib-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, required=True, help="Positive explicit limit; no implicit full-dataset run")
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-tokens", type=int, default=0)
    parser.add_argument("--kernel-hidden", choices=("rbf", "linear"), default="rbf")
    parser.add_argument("--kernel-y", choices=("rbf", "linear"), default="linear")
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--lx", type=float, default=1.0)
    parser.add_argument("--ly", type=float, default=1.0)
    parser.add_argument("--use-gpu-hsic", action="store_true")
    parser.add_argument("--no-activation-cache", action="store_true", help="Keep full-scale activations in RAM without writing a large cache")
    parser.add_argument("--checkpoint-every", type=int, default=0, help="Save exact Fisher accumulators and RNG every N samples; 0 disables")
    parser.add_argument("--resume", action="store_true", help="Resume own Fisher snapshot from the existing output directory")
    args = parser.parse_args()
    if args.checkpoint_every < 0 or (args.resume and args.stage != "fisher"):
        parser.error("Checkpoint interval must be nonnegative; resume supports Fisher only")
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    calib = load_raw(args.calib_dir)
    total = len(calib["state"])
    if not 0 < args.num_samples <= total:
        parser.error(f"--num-samples must be in [1,{total}]")
    if args.stage == "hsic" and args.num_samples < 2:
        parser.error("HSIC requires at least two different samples; four is only a smoke dataset")
    indices = np.sort(np.random.RandomState(args.seed).choice(total, args.num_samples, replace=False))
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if args.resume:
        if not np.array_equal(np.load(args.output / "indices.npy"), indices):
            raise ValueError("Resume sample indices differ")
    else:
        np.save(args.output / "indices.npy", indices)
    fisher, hsic = official_modules()
    report = {"status": "RUNNING", "stage": args.stage, "contract": experiment_contract(), "batch_size": 1,
              "num_samples": args.num_samples, "data_metadata": calib["meta"], "seed": args.seed,
              "checkpoint": str(args.checkpoint.resolve()), "calib_dir": str(args.calib_dir.resolve()),
              "source_sha256": {str(Path(module.__file__).resolve()): file_hash(Path(module.__file__))
                                for module in (fisher, hsic)}}
    report["runner_sha256"] = file_hash(Path(__file__))
    if args.resume:
        previous = json.loads((args.output / "run.json").read_text())
        for key in ("stage", "contract", "num_samples", "seed", "checkpoint", "calib_dir", "source_sha256", "runner_sha256"):
            if previous[key] != report[key]:
                raise ValueError(f"Resume contract differs: {key}")
        if previous.get("status") == "PASS":
            raise ValueError("Completed output cannot be resumed")
    def save():
        temporary = args.output / "run.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output / "run.json")
    start = time.monotonic()
    try:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        adapter = InputAdapter(args.checkpoint, args.tokenizer)
        model, config, aliases = load_calibration_model(args.checkpoint)
        report.update(strict_load="PASS", tied_aliases=aliases, runtime_precision=precision_snapshot())
        save()
        if args.stage == "hsic":
            for param in model.parameters():
                param.requires_grad_(False)
            original = hsic.build_observation_and_actions
            hsic.build_observation_and_actions = adapter.official_builder
            try:
                X, Y, inputs, outputs = hsic.collect_activations_pi05(model, calib, indices, None, None,
                    torch.device("cuda"), 50, 32, 32, 200, 1, args.pool_tokens)
            finally:
                hsic.build_observation_and_actions = original
            if not args.no_activation_cache:
                np.savez(args.output / "activations.npz", X_acts=X.astype(np.float16), Y_acts=Y.astype(np.float16),
                         in_acts=np.array({k: v.astype(np.float16) for k, v in inputs.items()}, dtype=object),
                         out_acts=np.array({k: v.astype(np.float16) for k, v in outputs.items()}, dtype=object))
            report.update(phase="scoring", activation_cache_saved=not args.no_activation_cache)
            save()
            del model
            gc.collect()
            torch.cuda.empty_cache()
            score_fn = hsic.compute_tensor_hsic_gpu if args.use_gpu_hsic else hsic.compute_tensor_hsic_cpu
            scores = score_fn(X, Y, inputs, outputs, kernel_hidden=args.kernel_hidden, kernel_y=args.kernel_y,
                              sigma=args.sigma, lx=args.lx, ly=args.ly)
            numeric_keys = ("hsic_x_out", "hsic_y_out", "F_out", "hsic_x_in", "hsic_y_in", "F_in", "sens")
            if not all(np.isfinite(item[key]) for item in scores["tensors"].values() for key in numeric_keys):
                raise ValueError("Non-finite HSIC output")
            scores["metadata"] = {**report, "pool_tokens": args.pool_tokens, "kernel_hidden": args.kernel_hidden,
                                  "kernel_y": args.kernel_y, "sigma": args.sigma, "lx": args.lx, "ly": args.ly}
            (args.output / "scores.json").write_text(json.dumps(scores, indent=2)+"\n")
            report["matrices"] = len(scores["tensors"])
        else:
            targets = fisher.get_target_params(model)
            fisher.freeze_non_target(model, targets)
            accumulators = fisher.init_accumulators(targets)
            losses = []
            offset = 0
            snapshot = args.output / "fisher-resume.pt"
            if args.resume:
                state = torch.load(snapshot, map_location="cpu", weights_only=False)
                if state["runner_sha256"] != report["runner_sha256"] or not np.array_equal(state["indices"], indices):
                    raise ValueError("Snapshot source or samples differ")
                if set(state["accumulators"]) != set(accumulators):
                    raise ValueError("Snapshot target set differs")
                for name in accumulators:
                    if state["accumulators"][name].shape != accumulators[name].shape:
                        raise ValueError(f"Snapshot shape differs: {name}")
                accumulators = state.pop("accumulators")
                losses = state["losses"]
                offset = len(losses)
                np.random.set_state(state["numpy_rng"])
                torch.set_rng_state(state["torch_rng"])
                torch.cuda.set_rng_state_all(state["cuda_rng"])
                del state
                report["resumed_samples"] = offset
            for step in range(offset, len(indices)):
                index = indices[step]
                obs, actions = adapter.build(calib, np.array([index]), "cuda")
                loss = fisher.fisher_step(model, targets, accumulators, obs, actions)
                if not np.isfinite(loss):
                    raise ValueError("Non-finite Fisher loss")
                losses.append(loss)
                print(f"Fisher sample {step+1}/{len(indices)}: loss={loss:.8f}", flush=True)
                del obs, actions
                torch.cuda.empty_cache()
                report.update(completed_samples=len(losses), duration_seconds=time.monotonic()-start)
                if len(losses) % 10 == 0 or len(losses) == len(indices):
                    save()
                if args.checkpoint_every and (len(losses) % args.checkpoint_every == 0 or len(losses) == len(indices)):
                    temporary = snapshot.with_suffix(".tmp")
                    torch.save({"runner_sha256": report["runner_sha256"], "indices": indices,
                                "accumulators": accumulators, "losses": losses, "numpy_rng": np.random.get_state(),
                                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, temporary)
                    temporary.replace(snapshot)
                    report["checkpoint_samples"] = len(losses)
                    save()
            llm, vision = {}, {}
            for name in list(accumulators):
                accumulator = accumulators.pop(name).numpy()
                if not np.isfinite(accumulator).all():
                    raise ValueError(f"Non-finite Fisher accumulator: {name}")
                array = (accumulator.astype(np.float64)/len(indices)).astype(np.float32)
                if name.startswith("v."):
                    vision[name] = pad_vision_for_export(name, array)
                else:
                    llm[name] = array
            fisher.write_imatrix_gguf(llm, args.output / "fisher_llm.gguf", len(indices), str(args.calib_dir))
            fisher.write_imatrix_gguf(vision, args.output / "fisher_vision.gguf", len(indices), str(args.calib_dir))
            np.save(args.output / "losses.npy", np.asarray(losses, dtype=np.float32))
            report.update(llm_matrices=len(llm), vision_matrices=len(vision), loss_mean=float(np.mean(losses)))
            if snapshot.exists():
                snapshot.unlink()  # This run's recovery state is superseded by its finished GGUF outputs.
        report.update(status="PASS", duration_seconds=time.monotonic()-start,
                      peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
    except Exception as error:
        report.update(status="FAIL", error=repr(error), duration_seconds=time.monotonic()-start)
        save()
        raise
    save()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

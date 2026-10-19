#!/usr/bin/env python3
"""Prepare one fixed sample and FP32 compute reference for exported FP16 weights."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from calibration_adapter import ROOT, load_calibration_model
from openpi_compare import observation_from_arrays
from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT / "reproduction/results/pi05-real-forward-01")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(4)
    start = time.monotonic()
    with np.load(args.reference / "inputs_and_reference.npz") as data:
        arrays = {key: data[key] for key in data.files}
    official_report = json.loads((args.reference / "report.json").read_text())
    (args.output / "prompt.txt").write_text(official_report["prompt"])
    model, _, aliases = load_calibration_model(ROOT / "models/pi05_libero")
    rope = model.paligemma_with_expert.paligemma.model.language_model.rotary_emb.inv_freq
    rope.detach().float().cpu().numpy().tofile(args.output / "rope_inv_freq.f32")
    obs, actions, noise, moment = observation_from_arrays(arrays, "cuda")
    report = {"scope": "export_precision_and_compacted_valid_prefix_reference_not_original_bf16_baseline",
              "strict_load": "PASS", "aliases": aliases,
              "sample": str(args.reference.resolve()), "dtype": "FP32 compute; rank>=2 weights rounded through FP16",
              "mask_compaction": "omit masked image and right-padded text tokens; keep valid token order"}
    signed = {}
    handle = model.action_out_proj.register_forward_hook(lambda module, inputs, output: signed.update(value=output.detach().float()))
    with torch.no_grad():
        # Validate original model path still reproduces the saved probe.
        initial_loss = model(obs, actions, noise=noise, time=moment)
        np.testing.assert_array_equal(initial_loss.cpu().numpy(), arrays["per_element_loss"])
        original_velocity = signed["value"].cpu().numpy()
        # Same precision transformation as official unified exporter.
        model.float()
        for param in model.parameters():
            if param.ndim >= 2:
                param.copy_(param.half().float())
        full_loss = model(obs, actions, noise=noise, time=moment)
        full_velocity = signed["value"].clone()
        n_tokens = int(obs.tokenized_prompt_mask.sum())
        obs.images = {key: value for key, value in obs.images.items() if bool(obs.image_masks[key].all())}
        obs.image_masks = {key: value for key, value in obs.image_masks.items() if bool(value.all())}
        obs.tokenized_prompt = obs.tokenized_prompt[:, :n_tokens]
        obs.tokenized_prompt_mask = obs.tokenized_prompt_mask[:, :n_tokens]
        # The official preprocessor requires three named cameras. The comparison
        # above already validated preprocessed inputs; bypass only that key-set
        # requirement for this compacted-input mathematical equivalence probe.
        def compact_preprocess(observation, *, train=True):
            return (list(observation.images.values()), list(observation.image_masks.values()),
                    observation.tokenized_prompt, observation.tokenized_prompt_mask, observation.state)
        model._preprocess_observation = compact_preprocess
        compact_loss = model(obs, actions, noise=noise, time=moment)
        compact_velocity = signed["value"].clone()
        images, masks, tokens, token_masks, state = model._preprocess_observation(obs, train=False)
        prefix, prefix_mask, _ = model.embed_prefix(images, masks, tokens, token_masks)
        if not bool(prefix_mask.all()) or prefix.shape != (1, 512+n_tokens, 2048):
            raise ValueError("Unexpected compacted prefix")
        time_expanded = moment[:, None, None]
        noisy_actions = time_expanded * noise + (1-time_expanded) * actions
        sinusoidal = create_sinusoidal_pos_embedding(moment, 1024, 4e-3, 4.0, device=moment.device).float()
        tensors = {"prefix_reference": prefix, "noisy_actions": noisy_actions,
                   "timestep_embedding": sinusoidal, "time": moment,
                   "velocity_reference": compact_velocity}
        for i, image in enumerate(images):
            tensors[f"image{i}"] = image
            tensors[f"image_embedding_reference{i}"] = prefix[:, i*256:(i+1)*256]
        tensors["text_embedding_reference"] = prefix[:, 512:]
        for name, value in tensors.items():
            value.detach().float().cpu().numpy().tofile(args.output / (name+".f32"))
        tokens.detach().cpu().numpy().astype(np.int32).tofile(args.output / "tokens.i32")
        np.savez(args.output / "references.npz", original_bf16_velocity=original_velocity,
                 full_export_precision_velocity=full_velocity.cpu().numpy(), compact_export_precision_velocity=compact_velocity.cpu().numpy())
        delta = (full_velocity-compact_velocity).abs()
        report.update(original_reference_loss="EXACT_MATCH", valid_text_tokens=n_tokens, valid_prefix_tokens=512+n_tokens,
                      compaction_max_abs_diff=float(delta.max()), compaction_mean_abs_diff=float(delta.mean()),
                      compaction_strict_allclose=bool(torch.allclose(full_velocity, compact_velocity, atol=1e-6, rtol=1e-5)),
                      duration_seconds=time.monotonic()-start, peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
    handle.remove()
    report["status"] = "PASS"
    (args.output / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

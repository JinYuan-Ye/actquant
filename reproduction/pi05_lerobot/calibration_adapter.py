"""Pinned LeRobot input contract for the official ActQuant computational routines."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer
from openpi.shared.image_tools import resize_with_pad_torch

from openpi_compare import ROOT, CAMERAS, build_config, load_target
from runtime_precision import configure_matmul_precision


def official_modules():
    for relative in ("tools/fisher-diag", "tools/hsic", "gguf-py"):
        path = str(ROOT / "src/ActQuant" / relative)
        if path not in sys.path:
            sys.path.insert(0, path)
    import compute_fisher_pi05 as fisher
    import compute_hsic_pi05 as hsic
    return fisher, hsic


def load_raw(directory):
    directory = Path(directory)
    data = {key: np.load(directory / (filename + ".npy"), mmap_mode="r", allow_pickle=False)
            for key, filename in (("pixel_values", "pixel_values"), ("state", "state"), ("gt_actions", "gt_actions"))}
    data["prompts"] = json.loads((directory / "prompts.json").read_text())
    data["meta"] = json.loads((directory / "metadata.json").read_text())
    pixels, state, actions = data["pixel_values"], data["state"], data["gt_actions"]
    n = len(pixels)
    if n < 1 or pixels.dtype != np.uint8 or pixels.ndim != 5 or pixels.shape[1:] != (3, 256, 256, 3):
        raise ValueError("Need native RGB uint8 (N,3,256,256,3); old PIL224 inputs are rejected")
    if state.shape != (n, 8) or actions.shape != (n, 50, 7):
        raise ValueError("Expected state (N,8), actions (N,50,7)")
    if state.dtype != np.float32 or actions.dtype != np.float32:
        raise ValueError("Raw state/actions must be float32")
    if len(data["prompts"]) != n or data["meta"]["total_frames"] != n:
        raise ValueError("Raw sample counts disagree")
    if not all(isinstance(task, str) and task.strip() for task in data["prompts"]):
        raise ValueError("Invalid task text")
    if data["meta"].get("image_mode") != "native":
        raise ValueError("Metadata must declare image_mode=native")
    if data["meta"].get("camera_order") != list(CAMERAS):
        raise ValueError("Unexpected camera order")
    return data


class InputAdapter:
    def __init__(self, checkpoint=ROOT / "models/pi05_libero", tokenizer=ROOT / "models/paligemma_tokenizer"):
        checkpoint = Path(checkpoint)
        self.config = build_config(checkpoint)
        raw = json.loads((checkpoint / "policy_preprocessor.json").read_text())
        steps = raw["steps"]
        if [s["registry_name"] for s in steps] != [
            "rename_observations_processor", "to_batch_processor", "normalizer_processor",
            "pi05_prepare_state_tokenizer_processor_step", "tokenizer_processor", "device_processor"]:
            raise ValueError("Unsupported processor pipeline; audit before adapting")
        if steps[0]["config"]["rename_map"] or steps[3]["config"]:
            raise ValueError("Unexpected rename or state-token options")
        norm = steps[2]
        if norm["config"]["norm_map"] != {"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}:
            raise ValueError("Unexpected normalization contract")
        self.stats = load_file(str(checkpoint / norm["state_file"]), device="cpu")
        self.eps = norm["config"]["eps"]
        options = steps[4]["config"]
        if options["padding_side"] != "right" or options["task_key"] != "task" or options["max_length"] != 200:
            raise ValueError("Unexpected tokenizer contract")
        self.token_options = {key: options[key] for key in ("max_length", "padding", "padding_side", "truncation")}
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer), local_files_only=True)

    def build(self, calib, indices, device):
        indices = np.asarray(indices)
        if indices.ndim != 1 or indices.size == 0 or indices.dtype.kind not in "iu":
            raise ValueError("Expected a nonempty integer index array")
        if np.any(indices < 0) or np.any(indices >= len(calib["state"])):
            raise IndexError("Calibration index out of range")
        # Match official reference: normalize on CPU, then tokenize state and move to device.
        state = torch.tensor(np.array(calib["state"][indices]), dtype=torch.float32)
        actions = torch.tensor(np.array(calib["gt_actions"][indices]), dtype=torch.float32)
        for key, value in (("observation.state", state), ("action", actions)):
            if not torch.isfinite(value).all():
                raise ValueError(f"Non-finite raw {key}")
        state = (state - self.stats["observation.state.mean"]) / (self.stats["observation.state.std"] + self.eps)
        actions = (actions - self.stats["action.mean"]) / (self.stats["action.std"] + self.eps)
        padded_state = F.pad(state, (0, 32 - state.shape[-1]))
        bins = np.digitize(padded_state.numpy(), np.linspace(-1, 1, 257)[:-1]) - 1
        prompts = []
        for row, index in enumerate(indices):
            task = calib["prompts"][int(index)].strip().replace("_", " ").replace("\n", " ")
            prompts.append(f"Task: {task}, State: {' '.join(map(str, bins[row]))};\nAction: ")
        tokens = self.tokenizer(prompts, **self.token_options, return_tensors="pt")
        images, masks = {}, {}
        for camera, name in enumerate(CAMERAS[:2]):
            # Match LeRobot: uint8 -> BCHW float/255 on CPU, then GPU NHWC resize.
            raw = torch.tensor(np.array(calib["pixel_values"][indices, camera])).permute(0, 3, 1, 2).float()/255
            image = raw.to(device).permute(0, 2, 3, 1)
            image = resize_with_pad_torch(image, 224, 224) * 2.0 - 1.0
            # OpenPI's utility squeezes B=1 for NHWC; LeRobot retains this axis.
            if image.ndim == 3 and len(indices) == 1:
                image = image.unsqueeze(0)
            images[name] = image.permute(0, 3, 1, 2)
            masks[name] = torch.ones(len(indices), dtype=torch.bool, device=device)
        images[CAMERAS[2]] = torch.ones_like(images[CAMERAS[1]]) * -1
        masks[CAMERAS[2]] = torch.zeros(len(indices), dtype=torch.bool, device=device)
        observation = SimpleNamespace(images=images, image_masks=masks, state=state.to(device),
            tokenized_prompt=tokens["input_ids"].to(device), tokenized_prompt_mask=tokens["attention_mask"].bool().to(device),
            token_ar_mask=None, token_loss_mask=None)
        return observation, F.pad(actions, (0, 25)).to(device)

    def official_builder(self, calib, indices, sp, norm_stats, device,
                         action_horizon, action_dim, state_dim, max_token_len):
        if (action_horizon, action_dim, state_dim, max_token_len) != (50, 32, 32, 200):
            raise ValueError("Calibration caller requested incompatible dimensions")
        return self.build(calib, indices, device)


def load_calibration_model(checkpoint, device="cuda"):
    config = build_config(Path(checkpoint))
    model, aliases = load_target(Path(checkpoint), config, device)
    configure_matmul_precision("highest")
    return model, config, aliases


def experiment_contract():
    return {"checkpoint_route": "LeRobot with audited external input adaptation",
            "action_horizon": 50, "action_dim": 32, "token_length": 200,
            "normalization": "checkpoint MEAN_STD", "loss": "original ActQuant full32 mean",
            "float32_matmul_precision": "highest", "image_augmentation": "disabled to match official LeRobot input",
            "fisher_batch_size": 1, "raw_images": "native256; official224 resize",
            "official_statistical_and_quantization_functions_modified": False}

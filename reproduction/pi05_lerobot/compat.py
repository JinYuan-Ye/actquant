"""Input/checkpoint compatibility only. No HSIC, Fisher or quantizer changes.

Uses LeRobot's registered preprocessor and model implementation. Imports are
local so help and metadata inspection do not initialize a model.
"""
from pathlib import Path
from types import SimpleNamespace
import json


def load_config(checkpoint, device="cpu"):
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.configs.policies import PreTrainedConfig
    cfg = PreTrainedConfig.from_pretrained(str(checkpoint), local_files_only=True)
    if not isinstance(cfg, PI05Config):
        raise ValueError("Expected a pi05 checkpoint")
    # Runtime controls: no compilation or training checkpointing for the probe.
    cfg.device = device
    cfg.compile_model = False
    cfg.gradient_checkpointing = False
    cfg.validate_features()
    return cfg


def make_preprocessor(checkpoint, tokenizer_dir, device="cpu"):
    # Import registers pi05_prepare_state_tokenizer_processor_step.
    from lerobot.policies.pi05 import processor_pi05  # noqa: F401
    from lerobot.processor import PolicyProcessorPipeline, batch_to_transition, transition_to_batch
    return PolicyProcessorPipeline.from_pretrained(
        str(checkpoint), config_filename="policy_preprocessor.json",
        overrides={
            "device_processor": {"device": device},
            "tokenizer_processor": {"tokenizer_name": str(Path(tokenizer_dir).resolve())},
        },
        to_transition=batch_to_transition, to_output=transition_to_batch,
    )


def load_strict(checkpoint, device="cuda"):
    """Official model and key remapper; fail hard on every loading mismatch.

    The v0.4.4 from_pretrained implementation catches loading exceptions and
    can return an uninitialized model. This wrapper deliberately propagates
    errors, and only fills missing aliases of actually tied Parameter objects.
    """
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from safetensors.torch import load_file
    path = Path(checkpoint) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Download explicitly first: {path}")
    cfg = load_config(checkpoint, "cpu")
    policy = PI05Policy(cfg)
    original = load_file(str(path), device="cpu")
    fixed = policy._fix_pytorch_state_dict_keys(original, cfg)
    state = {k if k.startswith("model.") else "model." + k: v for k, v in fixed.items()}
    aliases = {}
    for name, parameter in policy.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    filled = []
    for names in aliases.values():
        present = [name for name in names if name in state]
        if present:
            for name in names:
                if name not in state:
                    state[name] = state[present[0]]
                    filled.append([name, present[0]])
    policy.load_state_dict(state, strict=True)
    del state, fixed, original
    policy.to(device).eval()
    print(json.dumps({"strict_load": "PASS", "tied_aliases_filled": filled,
                      "chunk_size": cfg.chunk_size, "dtype": cfg.dtype}))
    return policy


def synthetic_sample(cfg):
    """Deterministic synthetic input for interface checks, never calibration."""
    import torch
    h, w = cfg.image_resolution
    return {
        "observation.images.image": torch.zeros(1, 3, h, w),
        "observation.images.image2": torch.full((1, 3, h, w), 0.5),
        "observation.state": torch.linspace(-0.2, 0.2, 8).reshape(1, 8),
        "action": torch.linspace(-0.1, 0.1, cfg.chunk_size * 7).reshape(1, cfg.chunk_size, 7),
        "task": ["pick up the cup"],
    }


def calib_sample(directory, index, cfg):
    """One batch from ActQuant raw calibration files; no implicit tail padding."""
    import numpy as np
    import torch
    directory = Path(directory)
    pixels = np.load(directory / "pixel_values.npy", mmap_mode="r")
    state = np.load(directory / "state.npy", mmap_mode="r")
    actions = np.load(directory / "gt_actions.npy", mmap_mode="r")
    prompts = json.loads((directory / "prompts.json").read_text())
    if not 0 <= index < len(pixels):
        raise ValueError("Sample index out of range")
    if pixels.dtype != np.uint8 or pixels.ndim != 5 or pixels.shape[1] < 2 or pixels.shape[-1] != 3:
        raise ValueError("Expected uint8 pixels (N, cameras>=2, H, W, 3)")
    if state.shape[1:] != (8,) or actions.ndim != 3 or actions.shape[2] != 7:
        raise ValueError("Expected raw state (N,8) and actions (N,H,7)")
    if actions.shape[1] < cfg.chunk_size:
        raise ValueError(f"Need {cfg.chunk_size} real/calibration-padded steps, got {actions.shape[1]}")
    if not (len(pixels) == len(state) == len(actions) == len(prompts)):
        raise ValueError("Calibration arrays have inconsistent lengths")
    result = {
        "observation.state": torch.tensor(np.array(state[index:index+1]), dtype=torch.float32),
        "action": torch.tensor(np.array(actions[index:index+1, :cfg.chunk_size]), dtype=torch.float32),
        "task": [prompts[index]],
    }
    for camera, key in enumerate(("observation.images.image", "observation.images.image2")):
        result[key] = torch.tensor(np.array(pixels[index:index+1, camera])).permute(0,3,1,2).float()/255.0
    return result


def validate_preprocessing(raw, batch, checkpoint, cfg):
    """Independent check of scales, token prompt and shapes, without a model."""
    import numpy as np
    import torch
    from safetensors.torch import load_file
    config = json.loads((Path(checkpoint) / "policy_preprocessor.json").read_text())
    step = next(s for s in config["steps"] if s["registry_name"] == "normalizer_processor")
    if step["config"]["norm_map"].get("ACTION") != "MEAN_STD" or step["config"]["norm_map"].get("STATE") != "MEAN_STD":
        raise ValueError("This audit expects the pinned MEAN_STD checkpoint")
    stats = load_file(str(Path(checkpoint) / step["state_file"]))
    for name in ("action", "observation.state"):
        expected = (raw[name].cpu() - stats[name + ".mean"]) / (stats[name + ".std"] + step["config"]["eps"])
        torch.testing.assert_close(batch[name].cpu(), expected, rtol=1e-5, atol=1e-6)
    if batch["action"].shape != (1, cfg.chunk_size, 7):
        raise ValueError(f"Unexpected normalized action shape: {batch['action'].shape}")
    for key in ("observation.language.tokens", "observation.language.attention_mask"):
        if tuple(batch[key].shape) != (1, cfg.tokenizer_max_length):
            raise ValueError(f"Unexpected {key} shape: {batch[key].shape}")
    normalized = batch["observation.state"].cpu().numpy()
    normalized = np.pad(normalized, ((0,0),(0,cfg.max_state_dim-normalized.shape[-1])))
    bins = np.digitize(normalized, np.linspace(-1,1,257)[:-1])-1
    cleaned = raw["task"][0].strip().replace("_", " ").replace("\n", " ")
    expected_prompt = f"Task: {cleaned}, State: {' '.join(map(str,bins[0]))};\nAction: "
    if batch["task"] != [expected_prompt]:
        raise ValueError("Official processor prompt differs from the pinned expected format")
    for value in batch.values():
        if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
            raise ValueError("Non-finite preprocessed tensor")
    return {"normalization": "PASS", "prompt_format": "PASS", "prompt": expected_prompt,
            "action_shape": list(batch["action"].shape),
            "tokens_shape": list(batch["observation.language.tokens"].shape)}


def openpi_inputs_from_reference(policy, batch):
    """Export correctly processed inputs for later, separate OpenPI comparison.

    Calls official LeRobot image handling; no extra normalization or fake
    camera is inserted into the raw batch. No numerical equivalence is claimed.
    """
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    images, masks = policy._preprocess_images(batch)
    if len(images) != 3:
        raise ValueError("Expected two real cameras plus one masked camera")
    names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    obs = SimpleNamespace(
        images=dict(zip(names, images)), image_masks=dict(zip(names, masks)),
        state=batch["observation.state"],
        tokenized_prompt=batch[OBS_LANGUAGE_TOKENS],
        tokenized_prompt_mask=batch[OBS_LANGUAGE_ATTENTION_MASK].bool(),
        token_ar_mask=None, token_loss_mask=None,
    )
    return obs, policy.prepare_action(batch)


def validate_tokenizer(batch, checkpoint, tokenizer_dir):
    """Audit actual IDs/masks using the checkpoint's explicit tokenizer options."""
    import torch
    from transformers import AutoTokenizer
    config = json.loads((Path(checkpoint) / "policy_preprocessor.json").read_text())
    options = next(s["config"] for s in config["steps"]
                   if s["registry_name"] == "tokenizer_processor")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    # Gemma's default can be left padding; the checkpoint explicitly uses right.
    # Omitting padding_side shifts both IDs and masks even for identical text.
    expected = tokenizer(
        batch[options["task_key"]], max_length=options["max_length"],
        padding=options["padding"], padding_side=options["padding_side"],
        truncation=options["truncation"], return_tensors="pt",
    )
    torch.testing.assert_close(batch["observation.language.tokens"].cpu(),
                               expected["input_ids"], rtol=0, atol=0)
    torch.testing.assert_close(batch["observation.language.attention_mask"].cpu().bool(),
                               expected["attention_mask"].bool(), rtol=0, atol=0)
    return {"tokenizer_exact_match": "PASS", "valid_tokens": int(expected["attention_mask"].sum()),
            "padding_side": options["padding_side"], "tokenizer_default_padding_side": tokenizer.padding_side}

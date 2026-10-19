#!/usr/bin/env python3
"""User-run download/extraction of one real RLDS probe; never formal calibration."""
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = "openvla/modified_libero_rlds"
REVISION = "6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551"
BUILDER = "libero_spatial_no_noops/1.0.0"
SHARD = BUILDER + "/libero_spatial-train.tfrecord-00000-of-00016"
SHARD_BYTES = 118202941
SHARD_SHA256 = "6d6360559a4b71df3bcabd9b55a75451f04375449ca5267616cbc1c21528eac5"
DEFAULT_DATA = ROOT / "data/libero_probe_rlds"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(directory):
    # Must be set before huggingface_hub import; activation defaults to offline.
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[key] = "0"
    from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
    from download_assets import matches

    records = []
    for name in (BUILDER + "/dataset_info.json", BUILDER + "/features.json", SHARD):
        metadata = get_hf_file_metadata(hf_hub_url(REPO, name, repo_type="dataset", revision=REVISION))
        if name == SHARD and (metadata.size != SHARD_BYTES or metadata.etag.strip('"') != SHARD_SHA256):
            raise ValueError("Remote shard metadata differs from pinned size/hash")
        path = directory / name
        if path.exists():
            if not matches(path, metadata.etag):
                raise FileExistsError(f"Refusing to overwrite differing file: {path}")
        else:
            print(f"Downloading {name}: {metadata.size} bytes", flush=True)
            hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION, local_dir=directory)
        if not matches(path, metadata.etag):
            raise ValueError(f"Integrity check failed: {path}")
        records.append({"filename": name, "bytes": metadata.size, "etag": metadata.etag})
        print("Verified:", path, flush=True)
    print(json.dumps({"repo": REPO, "revision": REVISION, "files": records}, indent=2))


def load_exporter():
    path = ROOT / "src/ActQuant/tools/fisher-diag/get_pi05_calib_data.py"
    spec = importlib.util.spec_from_file_location("actquant_raw_exporter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def select_sample(features, frame, horizon, exporter, image_mode="actquant"):
    """Validate a complete episode, then select one unpadded action window."""
    import numpy as np
    from PIL import Image

    def required(key, kind):
        value = features.get(key)
        if value is None or value[0] != kind or len(value[1]) == 0:
            raise ValueError(f"Missing or invalid {key}; no zero substitution allowed")
        return value[1]

    base = required("steps/observation/image", "bytes")
    wrist = required("steps/observation/wrist_image", "bytes")
    languages = required("steps/language_instruction", "bytes")
    states = np.asarray(required("steps/observation/state", "floats"), dtype=np.float32)
    actions = np.asarray(required("steps/action", "floats"), dtype=np.float32)
    count = len(base)
    if len(wrist) != count or len(languages) != count or states.size != count * 8 or actions.size != count * 7:
        raise ValueError("Episode images, language, state and action lengths disagree")
    states, actions = states.reshape(count, 8), actions.reshape(count, 7)
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Non-finite raw state/action")
    if frame < 0 or horizon <= 0 or frame + horizon > count:
        raise ValueError(f"Need {horizon} actual consecutive actions from frame {frame}; episode has {count}")
    task = languages[frame].decode("utf-8").lower().strip()
    if not task or any(item.decode("utf-8").lower().strip() != task for item in languages):
        raise ValueError("Empty or changing instruction inside episode")
    if image_mode not in ("native", "actquant"):
        raise ValueError(f"Unknown image mode: {image_mode}")
    cameras, original_shapes = [], []
    for data in (base[frame], wrist[frame]):
        with Image.open(io.BytesIO(data)) as image:
            raw = np.asarray(image.convert("RGB"))
        original_shapes.append(list(raw.shape))
        # Native mode leaves resizing to the official LeRobot model.
        # Keep ActQuant's historical PIL path available for explicit comparisons.
        cameras.append(raw if image_mode == "native" else exporter.resize_uint8(raw))
    cameras.append(np.zeros_like(cameras[0]))
    return (np.stack(cameras)[None], states[frame:frame + 1],
            actions[None, frame:frame + horizon], task, count, original_shapes)


def extract(directory, output, episode, frame, image_mode="actquant"):
    import numpy as np
    from PIL import Image

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if episode < 0:
        raise ValueError("Episode index must be non-negative")
    path = directory / SHARD
    if not path.is_file():
        raise FileNotFoundError("Pinned shard missing; run --mode download yourself first")
    if path.stat().st_size != SHARD_BYTES or sha256(path) != SHARD_SHA256:
        raise ValueError("Local shard size/SHA256 mismatch")
    exporter, exporter_path = load_exporter()
    for index, features in enumerate(exporter.iter_episodes([path])):
        if index == episode:
            break
    else:
        raise ValueError(f"Episode {episode} not found in pinned shard")
    pixels, state, actions, task, count, shapes = select_sample(features, frame, 50, exporter, image_mode)
    metadata = {
        "sample_kind": "real_interface_probe_not_formal_calibration",
        "repo": REPO, "revision": REVISION, "shard": SHARD,
        "shard_sha256": SHARD_SHA256, "exporter_sha256": sha256(exporter_path),
        "episode_index_in_shard": episode, "episode_frames": count, "frame_index": frame,
        "total_frames": 1, "action_horizon": 50, "action_tail_padding": False,
        "camera_order": ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
        "right_wrist_is_zeros": True, "original_image_shapes": shapes,
        "image_mode": image_mode,
        "image_transform": ("Native decoded RGB; resize deferred to official LeRobot model; no additional rotation"
                            if image_mode == "native" else
                            "ActQuant PIL bilinear resize to 224; no additional rotation"),
        "state_transform": "none; raw RLDS 8D state",
        "action_transform": "none; raw RLDS 7D actions",
        "normalization": "deferred to checkpoint official preprocessor",
        "semantic_alignment": "pending comparison with checkpoint training data",
        "task": task,
        "pixel_values_shape": list(pixels.shape), "state_shape": list(state.shape),
        "gt_actions_shape": list(actions.shape), "raw_state": state[0].tolist(),
        "raw_action_min_per_dim": actions.min(axis=(0, 1)).tolist(),
        "raw_action_max_per_dim": actions.max(axis=(0, 1)).tolist(),
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, array in (("pixel_values", pixels), ("state", state), ("gt_actions", actions)):
        np.save(output / (name + ".npy"), array)
    (output / "prompts.json").write_text(json.dumps([task]) + "\n")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Left = base camera, right = wrist camera, exactly as stored in the probe.
    Image.fromarray(np.concatenate([pixels[0, 0], pixels[0, 1]], axis=1)).save(output / "cameras.png")
    print(json.dumps(metadata, indent=2))
    print("Saved:", output)
    print("This is one real interface probe, not the paper's 60-episode calibration set.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("download", "extract"), required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--episode", type=int, default=0, help="Index within the pinned shard")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--image-mode", choices=("native", "actquant"), default="actquant",
                        help="Use native for the official reference; actquant preserves the earlier PIL export")
    args = parser.parse_args()
    if args.mode == "download":
        download(args.data_dir)
    else:
        if args.output is None:
            parser.error("--output must name a new directory for extract")
        extract(args.data_dir, args.output, args.episode, args.frame, args.image_mode)


if __name__ == "__main__":
    main()

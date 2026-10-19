#!/usr/bin/env python3
"""Reuse official 10/10/10/30 episode sampling, retaining native RGB for LeRobot."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from prepare_real_probe import load_exporter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Complete four-suite RLDS root, already downloaded")
    parser.add_argument("--output", type=Path, required=True, help="New native calibration directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sparse-placeholder-camera", action="store_true", help="Leave the all-zero third camera as sparse file holes; values and shape stay identical")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    exporter, source = load_exporter()
    for dataset in exporter.SUITE_TO_DATASET.values():
        builder = exporter.find_builder_dir(args.data_dir / dataset)
        info = json.loads((builder / "dataset_info.json").read_text())
        train = next(split for split in info["splits"] if split["name"] == "train")
        shards = exporter.list_shards(args.data_dir, dataset)
        if len(shards) != len(train["shardLengths"]):
            raise ValueError(f"Incomplete suite {dataset}: have {len(shards)} shards, need {len(train['shardLengths'])}")
    def retain_native(image):
        if image.shape != (256, 256, 3) or image.dtype != np.uint8:
            raise ValueError("Expected native256 RGB from the pinned RLDS source")
        return image
    # Change only image representation in this imported module instance.
    # Stratified episode selection, seed, frame selection, and tail-repeat remain upstream.
    exporter.resize_uint8 = retain_native
    exporter.IMAGE_SIZE = 256
    selections = []
    original_select = exporter.stratified_select
    def record_select(by_task, n_episodes, seed):
        selected, ep_to_task = original_select(by_task, n_episodes, seed)
        selections.append({"suite": list(exporter.SUITE_TO_DATASET)[len(selections)],
                           "selected_episode_indices": sorted(selected),
                           "tasks": {task: [i for i in sorted(selected) if ep_to_task[i] == task] for task in sorted(by_task)}})
        return selected, ep_to_task
    exporter.stratified_select = record_select
    original_save = np.save
    def save_with_sparse_camera(path, array, *positional, **keywords):
        if Path(path).name != "pixel_values.npy" or not args.sparse_placeholder_camera:
            return original_save(path, array, *positional, **keywords)
        if array.shape[1:] != (3, 256, 256, 3) or np.any(array[:, 2]):
            raise ValueError("Only the known all-zero placeholder can be stored as holes")
        mapped = np.lib.format.open_memmap(path, mode="w+", dtype=array.dtype, shape=array.shape)
        for start in range(0, len(array), 32):
            mapped[start:start+32, :2] = array[start:start+32, :2]
        mapped.flush()
        del mapped
    previous = sys.argv
    try:
        np.save = save_with_sparse_camera
        sys.argv = [str(source), "--data-dir", str(args.data_dir), "--output-dir", str(args.output),
                    "--seed", str(args.seed), "--action-horizon", "50"]
        exporter.main()
    finally:
        sys.argv = previous
        np.save = original_save
    path = args.output / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata.update(image_mode="native", action_tail_padding="repeat_last_as_in_official_exporter",
                    sample_kind="official_stratified_60_episodes_with_native_images",
                    source_exporter=str(source), source_exporter_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    episode_selections=selections, sparse_placeholder_camera=args.sparse_placeholder_camera,
                    adaptation="Only bypass PIL224 resize; model performs official resize; optional sparse zero camera preserves all values")
    path.write_text(json.dumps(metadata, indent=2)+"\n")


if __name__ == "__main__":
    main()

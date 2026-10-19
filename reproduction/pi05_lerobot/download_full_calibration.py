#!/usr/bin/env python3
"""Download and verify all four pinned LIBERO RLDS suites, resuming existing files."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

REPO = "openvla/modified_libero_rlds"
REVISION = "6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551"
SUITES = ("libero_spatial_no_noops", "libero_object_no_noops", "libero_goal_no_noops", "libero_10_no_noops")


def digest(path, git=False):
    h = hashlib.sha1() if git else hashlib.sha256()
    if git:
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[key] = "0"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import HfApi, hf_hub_download
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    items = list(HfApi().list_repo_tree(REPO, repo_type="dataset", revision=REVISION, recursive=True))
    files = []
    for item in items:
        if not hasattr(item, "size") or item.path.split("/")[0] not in SUITES:
            continue
        if not (".tfrecord-" in item.path or item.path.endswith(("/dataset_info.json", "/features.json"))):
            continue
        files.append({"path": item.path, "bytes": item.size,
                      "sha256": item.lfs["sha256"] if item.lfs else None,
                      "git_blob": item.blob_id if not item.lfs else None})
    files.sort(key=lambda x: x["path"])
    if len(files) != 104:
        raise ValueError(f"Expected 96 shards and 8 metadata files, got {len(files)}")
    missing_bytes = sum(f["bytes"] for f in files if not (args.data_dir / f["path"]).exists())
    if shutil.disk_usage(args.data_dir).free < missing_bytes + 4*1024**3:
        raise ValueError("Not enough disk space for RLDS download and reserve")
    report = {"status": "DOWNLOADING", "repo": REPO, "revision": REVISION,
              "total_bytes": sum(f["bytes"] for f in files), "files": files, "verified": []}

    def save():
        temporary = args.report.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2)+"\n")
        temporary.replace(args.report)

    def fetch(item):
        path = args.data_dir / item["path"]
        if not path.exists():
            for attempt in range(4):
                try:
                    hf_hub_download(REPO, item["path"], repo_type="dataset", revision=REVISION, local_dir=args.data_dir)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(min(2**attempt, 8))
        if path.stat().st_size != item["bytes"] or digest(path, git=item["sha256"] is None) != (item["sha256"] or item["git_blob"]):
            raise ValueError(f"Integrity mismatch; refusing overwrite: {path}")
        return item["path"]

    save()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch, item) for item in files]
            for future in as_completed(futures):
                path = future.result()
                report["verified"].append(path)
                save()
                print(f"Verified {len(report['verified'])}/{len(files)}: {path}", flush=True)
        report["status"] = "PASS"
        save()
    except Exception as error:
        report.update(status="FAIL", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()

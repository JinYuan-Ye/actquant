#!/usr/bin/env python3
"""Explicit user-run downloads, pinned revisions, no datasets or implicit weights."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = "lerobot/pi05_libero_finetuned_v044"
MODEL_REV = "7152f13e34c6c115d808aea0e5d08aca74eac14b"
TOKENIZER = "google/paligemma-3b-pt-224"
TOKENIZER_REV = "35e4f46485b4d07967e7e9935bc3786aad50687c"
METADATA = ["config.json", "policy_preprocessor.json", "policy_postprocessor.json",
            "policy_preprocessor_step_2_normalizer_processor.safetensors",
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors"]
TOKENIZER_FILES = {"tokenizer.model", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "added_tokens.json", "config.json"}


def matches(path, etag):
    """HF LFS etag is SHA256; regular Git blobs use SHA1 with a blob header."""
    etag = etag.strip('"')
    if len(etag) == 64:
        digest = hashlib.sha256()
    elif len(etag) == 40:
        digest = hashlib.sha1()
        digest.update(f"blob {path.stat().st_size}\0".encode())
    else:
        raise ValueError(f"Unsupported etag format for {path.name}")
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest() == etag


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", action="store_true")
    p.add_argument("--tokenizer", action="store_true", help="Small tokenizer files; may require HF access approval")
    p.add_argument("--weights", action="store_true", help="Explicitly download ~6.96 GiB model weights")
    args = p.parse_args()
    if not (args.metadata or args.tokenizer or args.weights):
        p.print_help(); return
    # Network is enabled only in this download process, before importing HF.
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "0"
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_download, hf_hub_url
    jobs = []
    if args.metadata or args.weights:
        files = METADATA + (["model.safetensors"] if args.weights else [])
        jobs.append((MODEL, MODEL_REV, ROOT/"models/pi05_libero", files))
    if args.tokenizer:
        available = set(HfApi().list_repo_files(TOKENIZER, revision=TOKENIZER_REV))
        files = sorted(available & TOKENIZER_FILES)
        if not {"tokenizer.model", "tokenizer_config.json"} <= set(files):
            raise RuntimeError("Pinned tokenizer repo missing required files")
        jobs.append((TOKENIZER, TOKENIZER_REV, ROOT/"models/paligemma_tokenizer", files))
    records = []
    for repo, rev, directory, files in jobs:
        directory.mkdir(parents=True, exist_ok=True)
        for name in files:
            path = directory/name
            meta = get_hf_file_metadata(hf_hub_url(repo, name, revision=rev))
            if path.exists():
                if not matches(path, meta.etag):
                    raise FileExistsError(f"Existing file differs; refusing to overwrite: {path}")
                status = "verified_existing"
            else:
                print(f"Downloading {repo}/{name}: {meta.size} bytes", flush=True)
                hf_hub_download(repo, name, revision=rev, local_dir=directory)
                if not matches(path, meta.etag):
                    raise RuntimeError(f"Integrity check failed: {path}")
                status = "downloaded_verified"
            records.append(dict(repo=repo, revision=rev, filename=name, size=meta.size,
                                etag=meta.etag, status=status))
            print(status, path, flush=True)
    logs = ROOT/"reproduction/logs"
    logs.mkdir(parents=True, exist_ok=True)
    output = logs / ("assets-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")+".json")
    with output.open("x") as f:
        json.dump(records, f, indent=2)
    print("Download manifest:", output)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Import only after main: importing HF before its offline override would
        # cache HF_HUB_OFFLINE=1 and prevent this explicit download command.
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
        status = getattr(getattr(error, "response", None), "status_code", None)
        if isinstance(error, GatedRepoError) or (isinstance(error, HfHubHTTPError) and status in (401, 403)):
            raise SystemExit(
                "Hugging Face 访问被拒绝（401/403），尚不能下载所需文件。\n"
                "1. 登录浏览器并在 https://huggingface.co/google/paligemma-3b-pt-224 接受访问条款。\n"
                "2. 为同一账号创建 Read token：https://huggingface.co/settings/tokens\n"
                "3. 在当前 lerobot-reference 终端运行：HF_HUB_OFFLINE=0 hf auth login\n"
                "4. 检查登录：HF_HUB_OFFLINE=0 hf auth whoami，然后重试原下载命令。\n"
                "如已登录，检查账号的模型访问权限和 token 的受限仓库读取权限。\n"
                "登录时交互输入 token；不要将 token 发到聊天中。"
            ) from None
        raise

#!/usr/bin/env python3
"""Publish authorized code and split model assets only after full quantization passes."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from prepare_github_publication import ROOT, REPO, TAG, PINS, prepare

GH = ROOT / ".local/bin/gh"


def load(path):
    return json.loads(path.read_text())


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def gh(*arguments):
    result = subprocess.run([str(GH), *map(str, arguments)], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"GitHub command failed: {' '.join(map(str, arguments[:3]))}: {result.stderr.strip()}")
    return result.stdout


def api(endpoint):
    return json.loads(gh("api", endpoint))


def split_model(model, assets, expected):
    assets.mkdir(parents=True, exist_ok=True)
    full = hashlib.sha256()
    parts = []
    with model.open("rb") as stream:
        index = 1
        while True:
            first = stream.read(8*1024**2)
            if not first:
                break
            target = assets / f"pi05.gguf.part{index:03d}"
            temporary = target.with_suffix(target.suffix+".tmp")
            h = hashlib.sha256()
            count = 0
            with temporary.open("xb") as destination:
                chunk = first
                while chunk:
                    destination.write(chunk)
                    h.update(chunk)
                    full.update(chunk)
                    count += len(chunk)
                    if count == 1024**3:
                        break
                    chunk = stream.read(min(8*1024**2, 1024**3-count))
            digest = h.hexdigest()
            if target.exists():
                if target.stat().st_size != count or hash_file(target) != digest:
                    raise ValueError(f"Existing model part differs: {target.name}")
                temporary.unlink()
            else:
                temporary.replace(target)
            parts.append({"filename": target.name, "bytes": count, "sha256": digest})
            index += 1
    if full.hexdigest() != expected:
        raise ValueError("Model changed since validated full-pipeline output")
    manifest = {"model_filename": "pi05.gguf", "model_bytes": model.stat().st_size,
                "model_sha256": expected, "parts": parts}
    (assets / "MODEL_SHA256.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, default=ROOT / "reproduction/results/pi05-full-01/pipeline.json")
    parser.add_argument("--staging", type=Path, default=ROOT / "exports/github-actquant-full-01")
    parser.add_argument("--state", type=Path, default=ROOT / "exports/github-publication-full-01.json")
    args = parser.parse_args()
    args.state.parent.mkdir(parents=True, exist_ok=True)
    lock = args.state.with_suffix(".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous_state = load(args.state) if args.state.exists() else {}
    if previous_state.get("status") == "PUBLISHED":
        print("Publication already completed; preserving it", flush=True)
        return
    state = {"status": "WAITING_FOR_FULL_QUANTIZATION", "repo": REPO, "tag": TAG, "pid": os.getpid(), "uploads_verified": []}

    def save():
        temporary = args.state.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2)+"\n")
        temporary.replace(args.state)

    save()
    try:
        permission = api(f"repos/{REPO}")
        if not permission.get("permissions", {}).get("push"):
            raise PermissionError("Authenticated account cannot push to authorized repository")
        while True:
            pipeline = load(args.pipeline)
            if pipeline["status"] == "PASS":
                state["pipeline_phase"] = pipeline["phase"]
                break
            if pipeline["status"] != "RUNNING":
                raise RuntimeError(f"Full pipeline is {pipeline['status']}; publishing is blocked")
            state["pipeline_phase"] = pipeline["phase"]
            save()
            time.sleep(15)
        if pipeline["calibration_episodes"] != 60 or pipeline["calibration_frames"] != 11777:
            raise ValueError("Unexpected full calibration coverage")
        if any(item["status"] != "PASS" for item in pipeline["stages"].values()):
            raise ValueError("Some full pipeline stage has not passed")
        state["status"] = "PREPARING_VERIFIED_RELEASE"
        save()
        state["source_preparation"] = prepare(args.staging, final=True)
        model = Path(pipeline["model"])
        assets = ROOT / "exports/github-full-model-assets-01"
        manifest = split_model(model, assets, pipeline["model_sha256"])
        for path in model.parent.iterdir():
            if path.is_file() and path != model:
                shutil.copy2(path, assets / path.name)
        shutil.copy2(ROOT / "deployment/restore_release_model.py", assets / "restore_release_model.py")
        audit = load(model.parent.parent / "artifact-audit.json")
        paired = load(args.pipeline.parent / "paired-validation.json")
        notes = f"""Full 60-episode calibration of the README-selected LeRobot π₀.₅ checkpoint.

- Calibration: all 11,777 frames; spatial/object/goal/long = 10/10/10/30 episodes.
- Quantized targets: all126 LLM matrices; vision, embeddings and action expert retain base precision.
- Actual target-matrix bpw: {audit['actual_block_bpw']:.9f}.
- Complete GGUF: {model.stat().st_size:,} bytes; SHA256 {pipeline['model_sha256']}.
- Artifact audits and paired same-backend ten-step execution passed.
- Single-observation denormalized7d action RMSE: {paired['sampled_actions_denormalized_7d']['rmse']:.8f}.
- Task quality NOT_EVALUATED; no LIBERO success-rate claim. Earlier strict cross-backend mismatch is retained.

Download all assets into one directory, then run `python restore_release_model.py --directory /path/to/assets`. This verifies each part and the reconstructed pi05.gguf. Splits exist because each GitHub Release asset must be smaller than2GiB. Tokenizer, normalization stats and runtime contract are included.

Model source: https://huggingface.co/lerobot/pi05_libero_finetuned_v044/tree/7152f13e34c6c115d808aea0e5d08aca74eac14b . Tokenizer source: https://huggingface.co/google/paligemma-3b-pt-224/tree/35e4f46485b4d07967e7e9935bc3786aad50687c . Source-code MIT license and upstream model provenance are separate; consult the model sources for their terms.
"""
        notes_path = assets / "MODEL_CARD.md"
        notes_path.write_text(notes)
        (assets / "SHA256SUMS.txt").write_text("".join(f"{hash_file(p)}  {p.name}\n" for p in sorted(assets.iterdir()) if p.is_file() and p.name != "SHA256SUMS.txt"))
        # Verify any final source preparation before external mutation.
        staged = args.staging.resolve()
        def git(*arguments, **kwargs):
            return subprocess.run(["git", "-C", str(staged), *arguments], check=True, **kwargs)
        user = api("user")
        git("config", "user.name", user.get("name") or user["login"])
        git("config", "user.email", f"{user['id']}+{user['login']}@users.noreply.github.com")
        git("add", "--all")
        for name in ("openpi", "lerobot-v044"):
            git("update-index", "--add", "--cacheinfo", f"160000,{PINS[name][1]},src/{name}")
        changes = subprocess.run(["git", "-C", str(staged), "diff", "--cached", "--quiet"])
        if changes.returncode == 1:
            git("commit", "-m", "Add full 60-episode pi05 calibration, compatibility fixes and quantization results")
        elif changes.returncode != 0:
            raise ValueError("Unable to inspect prepared commit")
        commit = subprocess.check_output(["git", "-C", str(staged), "rev-parse", "HEAD"], text=True).strip()
        state.update(status="PUSHING_SOURCE", commit=commit, model_sha256=manifest["model_sha256"])
        save()
        git("push", "origin", "HEAD:main")
        releases = api(f"repos/{REPO}/releases?per_page=100")
        release = next((r for r in releases if r["tag_name"] == TAG), None)
        if release is None:
            gh("release", "create", TAG, "--repo", REPO, "--draft", "--target", commit,
               "--title", "Pi05 full 60-episode LLM quantization", "--notes-file", notes_path)
            for attempt in range(10):
                releases = api(f"repos/{REPO}/releases?per_page=100")
                release = next((r for r in releases if r["tag_name"] == TAG), None)
                if release is not None:
                    break
                time.sleep(1)
            if release is None:
                raise RuntimeError("Created Release is not visible yet; safe to retry publication")
        if release["target_commitish"] != commit:
            if (release["draft"] and not release["assets"] and
                release["author"]["login"] == user["login"] and
                release["target_commitish"] == previous_state.get("commit")):
                # The preceding attempt created an empty draft before source fixes.
                # Update only that owned, empty draft; published releases stay immutable.
                release = json.loads(gh("api", "--method", "PATCH", f"repos/{REPO}/releases/{release['id']}",
                                        "-f", "target_commitish="+commit))
            else:
                raise ValueError("Existing Release targets another commit; refusing replacement")
        state.update(status="UPLOADING_MODEL_ASSETS", release_id=release["id"])
        save()
        expected_assets = {path.name: path for path in assets.iterdir() if path.is_file()}
        if set(a["name"] for a in release["assets"]) - set(expected_assets):
            raise ValueError("Existing Release has unexpected assets; refusing to modify it")
        for name, path in sorted(expected_assets.items()):
            existing = next((a for a in release["assets"] if a["name"] == name), None)
            expected_hash = "sha256:"+hash_file(path)
            if existing:
                if existing["state"] != "uploaded" or existing["size"] != path.stat().st_size or existing.get("digest") != expected_hash:
                    raise ValueError(f"Existing Release asset differs: {name}")
            else:
                if not release["draft"]:
                    raise ValueError("Refusing to change an already published incomplete Release")
                gh("release", "upload", TAG, path, "--repo", REPO)
            current = api(f"repos/{REPO}/releases/{release['id']}")
            uploaded = next(a for a in current["assets"] if a["name"] == name)
            if uploaded["state"] != "uploaded" or uploaded["size"] != path.stat().st_size or uploaded.get("digest") != expected_hash:
                raise ValueError(f"Uploaded asset hash/size mismatch: {name}")
            state["uploads_verified"].append(name)
            save()
            print("Uploaded and verified:", name, flush=True)
        if release["draft"]:
            gh("release", "edit", TAG, "--repo", REPO, "--draft=false")
        published = api(f"repos/{REPO}/releases/{release['id']}")
        branch = api(f"repos/{REPO}/git/ref/heads/main")
        if published["draft"] or branch["object"]["sha"] != commit:
            raise ValueError("Remote publication verification failed")
        state.update(status="PUBLISHED", source_url=f"https://github.com/{REPO}/tree/{commit}", release_url=published["html_url"])
        save()
        print(json.dumps(state, indent=2), flush=True)
    except BaseException as error:
        state.update(status="FAIL", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()

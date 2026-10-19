#!/usr/bin/env python3
"""Push the authorized scripts/docs now, without publishing unfinished models."""
import json
from pathlib import Path
import subprocess

from prepare_github_publication import ROOT, REPO, PINS, prepare


def main():
    gh = ROOT / ".local/bin/gh"
    staging = ROOT / "exports/github-actquant-full-01"
    prepare(staging, final=False)
    user = json.loads(subprocess.check_output([str(gh), "api", "user"], text=True))

    def git(*args):
        return subprocess.run(["git", "-C", str(staging), *args], check=True, capture_output=True, text=True)

    git("config", "user.name", user.get("name") or user["login"])
    git("config", "user.email", f"{user['id']}+{user['login']}@users.noreply.github.com")
    git("add", "--all")
    for name in ("openpi", "lerobot-v044"):
        git("update-index", "--add", "--cacheinfo", f"160000,{PINS[name][1]},src/{name}")
    diff = subprocess.run(["git", "-C", str(staging), "diff", "--cached", "--quiet"])
    if diff.returncode == 1:
        git("commit", "-m", "Publish pi05 calibration and quantization scripts with compatibility documentation")
    elif diff.returncode:
        raise RuntimeError("Unable to inspect staged changes")
    commit = git("rev-parse", "HEAD").stdout.strip()
    git("push", "origin", "HEAD:main")
    remote = json.loads(subprocess.check_output([str(gh), "api", f"repos/{REPO}/git/ref/heads/main"], text=True))
    if remote["object"]["sha"] != commit:
        raise ValueError("Remote commit verification failed")
    report = {"status": "CODE_PUBLISHED", "commit": commit, "source_url": f"https://github.com/{REPO}/tree/{commit}",
              "scope": "scripts, documentation, pinned upstream source and small reports; no model weights"}
    (ROOT / "exports/code-publication-01.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

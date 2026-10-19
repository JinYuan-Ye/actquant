#!/usr/bin/env python3
"""Verify a published upstream source snapshot without requiring its own .git."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(directory, revision):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / ".upstream-snapshot.json").read_text())
    if manifest["revision"] != revision:
        raise ValueError("Upstream revision differs from the pinned installer")
    for name, expected in manifest["files"].items():
        path = directory / name
        if not path.resolve().is_relative_to(directory) or not path.is_file():
            raise ValueError(f"Missing or unsafe source snapshot file: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source snapshot file differs: {name}")
    return len(manifest["files"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    print(f"Verified {verify(args.source_dir, args.revision)} upstream snapshot files")

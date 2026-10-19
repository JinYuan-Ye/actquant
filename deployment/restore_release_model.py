#!/usr/bin/env python3
"""Reassemble verified GitHub Release model parts on Windows or Linux."""
import argparse
import hashlib
import json
from pathlib import Path


def restore(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "MODEL_SHA256.json").read_text())
    output = directory / manifest["model_filename"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    temporary = output.with_suffix(".gguf.tmp")
    overall = hashlib.sha256()
    total = 0
    with temporary.open("xb") as destination:
        for part in manifest["parts"]:
            name = part["filename"]
            if Path(name).name != name:
                raise ValueError("Model part must be a plain filename")
            source = directory / name
            if source.stat().st_size != part["bytes"]:
                raise ValueError(f"Part size mismatch: {name}")
            digest = hashlib.sha256()
            with source.open("rb") as stream:
                for block in iter(lambda: stream.read(8*1024**2), b""):
                    destination.write(block)
                    digest.update(block)
                    overall.update(block)
                    total += len(block)
            if digest.hexdigest() != part["sha256"]:
                raise ValueError(f"Part SHA256 mismatch: {name}")
    if total != manifest["model_bytes"] or overall.hexdigest() != manifest["model_sha256"]:
        raise ValueError("Reassembled model SHA256/size mismatch; partial output retained for inspection")
    temporary.replace(output)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    print(f"Verified complete model: {restore(args.directory)}")

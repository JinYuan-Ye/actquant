#!/usr/bin/env python3
"""Quantize all 126 LLM targets with explicit HSIC formats and per-weight Fisher."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, default=ROOT / "reproduction/results/pi05-allocation-demo-01/allocation.json")
    parser.add_argument("--imatrix", type=Path, default=ROOT / "reproduction/results/pi05-fisher-demo-01/fisher_llm.gguf")
    parser.add_argument("--input", type=Path, default=ROOT / "reproduction/results/pi05-quant-bridge-01/llm/pali_llm_bf16.gguf")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.allocation.read_text())
    expected = {f"blk.{i}.{kind}.weight" for i in range(18) for kind in
                ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")}
    if set(data["assignments"]) != expected or args.threads < 1:
        parser.error("Need exact 126 assignments and positive thread count")
    command = [str(ROOT / "build/pi05/bin/llama-quantize"), "--imatrix", str(args.imatrix.resolve()),
               "--token-embedding-type", "f16", "--leave-output-tensor"]
    for name, kind in sorted(data["assignments"].items()):
        command.extend(["--tensor-type", "^"+re.escape(name)+"$="+kind])
    command.extend([str(args.input.resolve()), str(args.output.resolve() / "pali_llm_mixed.gguf"), "IQ2_XS", str(args.threads)])
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "scope": "all126_LLM_matrices; quality depends on supplied calibration",
              "command": command, "allocation": str(args.allocation.resolve()),
              "allocation_sha256": hashlib.sha256(args.allocation.read_bytes()).hexdigest()}

    def save():
        (args.output / "run.json").write_text(json.dumps(report, indent=2)+"\n")

    save()
    start = time.monotonic()
    try:
        with (args.output / "quantize.log").open("x") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        report.update(status="QUANTIZATION_FINISHED" if not result.returncode else "FAIL",
                      returncode=result.returncode, seconds=time.monotonic()-start)
        save()
        if result.returncode:
            raise SystemExit(result.returncode)
    except Exception as error:
        report.update(status="FAIL", error=repr(error), seconds=time.monotonic()-start)
        save()
        raise
    print(json.dumps({k: v for k, v in report.items() if k != "command"}, indent=2))


if __name__ == "__main__":
    main()

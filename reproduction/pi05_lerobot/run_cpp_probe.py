#!/usr/bin/env python3
"""Build an external fixed-input probe and preserve execution/numerical statuses.

Uses existing pinned core objects and local model files. No downloads or policy
server. A successful execution with strict numerical mismatch exits with code 2.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "reproduction/results/pi05-quant-bridge-01/unified/pi05.gguf")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "models/paligemma_tokenizer/tokenizer.model")
    parser.add_argument("--mode", choices=("upstream", "f32_request", "no_tf32", "matched_rope"), default="matched_rope")
    parser.add_argument("--ode-steps", type=int, choices=range(0,21), default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    build = ROOT / "build/pi05"
    entries = json.loads((build / "compile_commands.json").read_text())
    entry = next(e for e in entries if e["file"].endswith("/tools/pi0.5/main.cpp"))
    flags = [arg for arg in shlex.split(entry["command"])[1:] if arg.startswith(("-D", "-I"))]
    objects = sorted((build / "tools/pi0.5/CMakeFiles/pi05_core.dir").glob("*.o"))
    if len(objects) != 8:
        raise ValueError("Unexpected core object list; rebuild pinned base first")
    source = ROOT / "reproduction/pi05_lerobot/cpp_forward_probe.cpp"
    libraries = [build / "common/libcommon.a", build / "tools/tokenizers/sentencepiece/src/libsentencepiece.a"]
    libraries += [build / "bin" / name for name in ("libllama.so", "libggml.so", "libggml-cpu.so", "libggml-cuda.so", "libggml-base.so")]
    binary = args.output.resolve() / "cpp-forward-probe"
    command = ["/usr/bin/c++", "-std=c++17", "-O2", "-DNDEBUG", *flags, str(source),
               *map(str, objects), *map(str, libraries), "-pthread", "-Wl,--export-dynamic", "-ldl",
               "-Wl,-rpath,"+str(build / "bin"), "-o", str(binary)]
    report = {"scope": "single_sample_diagnostic_not_validated_policy", "mode": args.mode,
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "build_command": command}
    start = time.monotonic()

    def save():
        (args.output / "run.json").write_text(json.dumps(report, indent=2)+"\n")

    with (args.output / "build.log").open("x") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    report["build_returncode"] = result.returncode
    save()
    if result.returncode:
        raise SystemExit(result.returncode)
    environment = dict(os.environ)
    for name in ("ACTQUANT_PROBE_F32", "ACTQUANT_PROBE_MATCH_ROPE", "ACTQUANT_PROBE_CAST_WEIGHTS"):
        environment.pop(name, None)
    if args.mode != "upstream":
        environment["ACTQUANT_PROBE_F32"] = "1"
    if args.mode in ("no_tf32", "matched_rope"):
        environment["NVIDIA_TF32_OVERRIDE"] = "0"
    if args.mode == "matched_rope":
        environment["ACTQUANT_PROBE_MATCH_ROPE"] = "1"
    environment.pop("ACTQUANT_PROBE_ODE_STEPS", None)
    if args.ode_steps:
        environment["ACTQUANT_PROBE_ODE_STEPS"] = str(args.ode_steps)
    command = list(map(str, [binary, args.model.resolve(), args.tokenizer.resolve(), args.reference.resolve(), args.output.resolve() / "outputs"]))
    report.update(command=command, environment={name: environment.get(name) for name in
                  ("ACTQUANT_PROBE_F32", "ACTQUANT_PROBE_MATCH_ROPE", "NVIDIA_TF32_OVERRIDE", "ACTQUANT_PROBE_ODE_STEPS")})
    with (args.output / "execution.log").open("x") as log:
        result = subprocess.run(command, cwd=build / "bin", env=environment, stdout=log, stderr=subprocess.STDOUT)
    report["execution_returncode"] = result.returncode
    save()
    if result.returncode:
        report.update(status="EXECUTION_FAILED", seconds=time.monotonic()-start)
        save()
        raise SystemExit(1)
    compare = ROOT / "reproduction/pi05_lerobot/compare_cpp_forward.py"
    with (args.output / "comparison.log").open("x") as log:
        result = subprocess.run([sys.executable, str(compare), "--reference", str(args.reference.resolve()),
                                 "--actual", str(args.output.resolve() / "outputs")], stdout=log, stderr=subprocess.STDOUT)
    status = "PASS" if not result.returncode else ("NUMERICAL_MISMATCH" if result.returncode == 2 else "COMPARISON_FAILED")
    report.update(comparison_returncode=result.returncode, status=status,
                  seconds=time.monotonic()-start)
    save()
    print(json.dumps({k: v for k, v in report.items() if k not in ("build_command", "command")}, indent=2))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

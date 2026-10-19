#!/usr/bin/env python3
"""Run the authorized 60-episode/all-frame LLM quantization with audited stages."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "reproduction/pi05_lerobot"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--calib-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/libero_probe_rlds")
    parser.add_argument("--wait-for-download", action="store_true", help="Wait for the separately tracked downloader")
    parser.add_argument("--fisher-execution", choices=("cpu", "gpu"), default="cpu", help="Validated equivalent accumulator storage backend")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    args.calib_dir = args.calib_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = run_dir / "pipeline.json"
    report = read(output) if output.exists() else {"status": "RUNNING", "stages": {}, "scope": "60 episodes, every selected frame; all126 LLM targets", "seed": 42}
    if report.get("status") == "PASS":
        raise ValueError("Completed pipeline cannot be overwritten")
    report.update(status="RUNNING", pid=os.getpid(), runner_sha256=sha(Path(__file__)),
                  authorized_scope="User explicitly requested full calibration-set quantization", calib_dir=str(args.calib_dir))

    def save():
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2)+"\n")
        temporary.replace(output)

    def stage(name, command, accepts=(0,), resume=False):
        old = report["stages"].get(name)
        if old and old.get("status") == "PASS":
            print(f"Keep finished stage: {name}", flush=True)
            return
        if old and not resume:
            raise ValueError(f"Unfinished {name} cannot overwrite its output; inspect stage before resuming")
        record = {"status": "RUNNING", "command": list(map(str, command)), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if old:
            record["previous_attempt"] = old
        report["stages"][name] = record
        report["phase"] = name
        save()
        print(f"Starting {name}", flush=True)
        started = time.monotonic()
        log_path = run_dir / (name+".log")
        with log_path.open("a" if old else "x") as log:
            process = subprocess.Popen(record["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            record["pid"] = process.pid
            save()
            while True:
                try:
                    code = process.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    record["seconds"] = time.monotonic()-started
                    save()
        record.update(returncode=code, seconds=time.monotonic()-started, status="PASS" if code in accepts else "FAIL")
        save()
        if code not in accepts:
            raise RuntimeError(f"Stage {name} failed ({code}); see {log_path}")
        print(f"Finished {name}: {record['seconds']:.1f}s", flush=True)

    def py(script, *arguments):
        return [sys.executable, "-u", SCRIPTS / script, *arguments]

    save()
    try:
        downloaded = run_dir / "download.json"
        if args.wait_for_download:
            report["phase"] = "download"
            save()
            start = time.monotonic()
            while True:
                state = read(downloaded) if downloaded.exists() else {}
                if state.get("status") == "PASS":
                    break
                if state.get("status") == "FAIL":
                    raise RuntimeError("Tracked download failed; inspect download.json")
                if time.monotonic()-start > 6*3600:
                    raise TimeoutError("Tracked download did not finish within six hours")
                time.sleep(10)
        elif not downloaded.exists() or read(downloaded).get("status") != "PASS":
            stage("download", py("download_full_calibration.py", "--data-dir", args.data_dir, "--report", downloaded), resume=True)
        if read(downloaded).get("status") != "PASS":
            raise ValueError("Full source download is not verified")
        stage("export", py("export_native_calibration.py", "--data-dir", args.data_dir,
                           "--output", args.calib_dir, "--sparse-placeholder-camera"))
        stage("data_audit", py("audit_full_calibration.py", "--calib-dir", args.calib_dir,
                               "--output", run_dir / "data-audit.json"))
        data_audit = read(run_dir / "data-audit.json")["data"]
        total = data_audit["frames"]
        report.update(calibration_frames=total, calibration_episodes=60, suites=data_audit["suites"])
        save()
        # Budget for two recovery snapshots (atomic replacement), both Fisher exports,
        # and the standalone/full quantized outputs. These do not coexist at peak.
        existing_snapshot = run_dir / "fisher/fisher-resume.pt"
        required = max(0, 24*1024**3 - (existing_snapshot.stat().st_size if existing_snapshot.exists() else 0))
        if shutil.disk_usage(run_dir).free < required:
            raise ValueError("Need at least 24 GiB free after data export for safe Fisher checkpoints and quantization")
        hsic_dir, fisher_dir = run_dir / "hsic", run_dir / "fisher"
        allocation = run_dir / "allocation.json"
        quant_dir = run_dir / "quantized"
        common = ["--calib-dir", args.calib_dir, "--num-samples", str(total)]
        stage("hsic", py("run_calibration.py", "--stage", "hsic", *common,
                         "--use-gpu-hsic", "--no-activation-cache", "--output", hsic_dir))
        score = read(hsic_dir / "scores.json")
        import numpy as np
        if score["n_samples"] != total or len(score["tensors"]) != 126 or not np.array_equal(np.load(hsic_dir / "indices.npy"), np.arange(total)):
            raise ValueError("HSIC does not cover all calibration frames and 126 matrices")
        stage("allocation", [sys.executable, ROOT / "src/ActQuant/tools/hsic/allocate_pi05.py",
                             "--scores", hsic_dir / "scores.json", "--score-key", "F_out", "--target-bpw", "2.3471",
                             "--base-type", "IQ2_XS", "--max-type", "Q4_K", "--output", allocation])
        assigned = read(allocation)
        report["allocation"] = {"target_bpw": assigned["target_bpw"], "achieved_bpw": assigned["achieved_bpw"],
                                "format_counts": dict(Counter(assigned["assignments"].values()))}
        save()
        resume_args = ["--resume"] if (fisher_dir / "fisher-resume.pt").exists() else []
        fisher_script = "run_fisher_gpu.py" if args.fisher_execution == "gpu" else "run_calibration.py"
        stage("fisher", py(fisher_script, "--stage", "fisher", *common, "--output", fisher_dir,
                           "--checkpoint-every", "200", *resume_args), resume=bool(resume_args))
        stage("fisher_audit", py("audit_full_calibration.py", "--calib-dir", args.calib_dir,
                                 "--fisher-dir", fisher_dir, "--output", run_dir / "fisher-audit.json"))
        stage("quantization", py("quantize_llm.py", "--allocation", allocation,
                                 "--imatrix", fisher_dir / "fisher_llm.gguf", "--output", quant_dir))
        stage("merge", py("finish_llm_quantization.py", "--run-dir", quant_dir, "--allocation", allocation,
                          "--imatrix", fisher_dir / "fisher_llm.gguf"))
        if not read(quant_dir / "artifact-audit.json")["full_calibration"]:
            raise ValueError("Final artifact is not marked as full calibration")
        reference = ROOT / "reproduction/results/pi05-cpp-reference-01"
        model = quant_dir / "model"
        shutil.copy2(reference / "rope_inv_freq.f32", model / "rope_inv_freq.f32")
        (model / "runtime_contract.json").write_text(json.dumps({"calibration": f"60 stratified episodes; all {total} frames",
            "validated_runtime": "external fixed-input CUDA probe; matched_rope; 10-step paired sampling",
            "valid_prefix_contract": "state-bearing prompt, two real cameras, compact valid prefix",
            "rope": "source BF16-rounded frequencies in rope_inv_freq.f32", "original_serve_policy_validated": False,
            "task_quality": "NOT_EVALUATED", "cross_backend_strict_status": "NUMERICAL_MISMATCH retained"}, indent=2)+"\n")
        for name, path in (("baseline", ROOT / "reproduction/results/pi05-quant-bridge-01/unified/pi05.gguf"),
                           ("quantized_forward", model / "pi05.gguf")):
            stage(name, py("run_cpp_probe.py", "--reference", reference, "--model", path,
                           "--mode", "matched_rope", "--ode-steps", "10", "--output", run_dir / name), accepts=(0, 2))
            if read(run_dir / name / "run.json")["execution_returncode"] != 0:
                raise ValueError(f"Failed execution: {name}")
        stage("paired_validation", py("measure_quantization_effect.py", "--baseline", run_dir / "baseline",
                                      "--quantized", run_dir / "quantized_forward", "--model-dir", model,
                                      "--output", run_dir / "paired-validation.json"))
        report.update(status="PASS", phase="finished", model=str(model / "pi05.gguf"),
                      model_sha256=sha(model / "pi05.gguf"), quality_acceptance="NOT_EVALUATED",
                      validation="artifact_and_single_observation_paired_execution; not_LIBERO_success")
        (model / "validation_manifest.json").write_text(json.dumps({k: v for k, v in report.items() if k != "stages"}, indent=2)+"\n")
        save()
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()

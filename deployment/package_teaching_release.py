#!/usr/bin/env python3
"""Archive the teaching code, documents, evidence and final model for Windows transfer."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODEL_SHA = "25d52c5e7a54608b604b15322ad09fa0b0502209bd988b45c313f54f8528b41e"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="pi05-teaching-20261019-01")
    args = parser.parse_args()
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.name):
        raise ValueError("Archive name must be a simple directory name")
    exports = ROOT / "exports"
    exports.mkdir(exist_ok=True)
    staging = exports / args.name
    staging.mkdir(exist_ok=False)
    archive = exports / (args.name + ".tar")
    if archive.exists():
        raise FileExistsError(archive)
    files = {}

    def add(path, destination=None):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Only regular files can be packaged: {path}")
        name = destination or path.relative_to(ROOT).as_posix()
        if name in files and files[name] != path:
            raise ValueError(f"Duplicate archive member: {name}")
        files[name] = path

    for path in sorted((ROOT / "reproduction/pi05_lerobot").iterdir()):
        if path.is_file() and path.suffix in (".py", ".cpp", ".md"):
            add(path)
    for path in sorted((ROOT / "deployment").iterdir()):
        if path.is_file() and path.suffix in (".py", ".sh", ".ps1", ".md", ".txt", ".lock", ".in"):
            add(path)
    evidence_names = (
        "openpi-server-freeze.txt", "openpi-libero-freeze.txt", "lerobot-reference-freeze.txt",
        "lerobot-reference-environment.json", "transformers-patch-manifest.json", "reference-sha256.txt",
    )
    for name in evidence_names:
        add(ROOT / "deployment/evidence" / name)
    add(ROOT / "handoff.md")
    for path in sorted((ROOT / "reproduction/evidence").glob("*.json")):
        add(path)
    for path in sorted((ROOT / "reproduction/data/pi05-readiness-01").iterdir()):
        if path.suffix in (".npy", ".json"):
            add(path)

    # Preserve small reports, including historical mismatches, without bundling large intermediates.
    for path in sorted((ROOT / "reproduction/results").rglob("*.json")):
        if not path.is_symlink() and path.stat().st_size < 2 * 1024 * 1024:
            add(path)
    selected_runs = (
        "pi05-real-forward-01", "pi05-cpp-reference-01", "pi05-quant-baseline-01",
        "pi05-quantized-forward-01", "pi05-quantized-demo-01", "pi05-hsic-demo-01", "pi05-fisher-demo-01",
    )
    for name in selected_runs:
        for path in sorted((ROOT / "reproduction/results" / name).rglob("*")):
            if path.is_file() and not path.is_symlink() and path.suffix in (".npz", ".npy", ".f32", ".i32", ".txt", ".log") and path.stat().st_size < 10 * 1024 * 1024:
                add(path)
    model_dir = ROOT / "reproduction/results/pi05-quantized-demo-01/model"
    for path in sorted(model_dir.iterdir()):
        if path.is_file():
            add(path)

    upstream = [
        "README.md", "src/llama-quant.cpp", "tools/hsic/allocate_pi05.py",
        "tools/hsic/compute_hsic_pi05.py", "tools/hsic/hsic_utils.py", "tools/hsic/run_hsic_quant_pi05.sh",
        "tools/fisher-diag/compute_fisher_pi05.py", "tools/fisher-diag/get_pi05_calib_data.py",
        "tools/pi0.5/export_pi05.py", "tools/pi0.5/export_pi05_llm.py", "tools/pi0.5/merge_pi05_llm.py",
    ]
    repo = ROOT / "src/ActQuant"
    for name in upstream:
        path = repo / name
        add(path, "upstream_reference/src/ActQuant/" + name)
    for path in sorted(repo.glob("LICENSE*")):
        if path.is_file():
            add(path, "upstream_reference/src/ActQuant/" + path.name)

    status = {}
    for name in ("ActQuant", "openpi", "lerobot-v044"):
        repo = ROOT / "src" / name
        def git(*arguments):
            return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True)
        status[name] = {"commit": git("rev-parse", "HEAD").strip(),
                        "status": git("status", "--short"),
                        "tracked_changes": git("diff", "HEAD", "--name-only").splitlines()}
        if status[name]["tracked_changes"]:
            raise ValueError(f"Unexpected upstream tracked changes: {name}")
    (staging / "SOURCE_STATUS.json").write_text(json.dumps(status, indent=2) + "\n")
    (staging / "README.md").write_text("""# ActQuant π₀.₅ 教学版交付

本包包含新增适配/量化代码、主要问题与解决记录、教学文档、四帧样本、小型验证证据，以及最终完整GGUF模型。

- 改动清单：deployment/TEACHING_RELEASE_CHANGES.md。
- 主要问题与解决：deployment/TEACHING_RELEASE_ISSUES.md。
- 教学入口：reproduction/pi05_lerobot/METHOD_WALKTHROUGH.md。
- 量化结果及复现命令：reproduction/pi05_lerobot/QUANTIZED_MODEL_DEMO.md。
- 模型：reproduction/results/pi05-quantized-demo-01/model/pi05.gguf。
- 实验结果：reproduction/results/pi05-quantized-demo-01/completion.json、paired-validation.json。

模型3,331,893,600字节。全部126个LLM目标量化，其他组件保持基础精度。本轮仅用同一轨迹四帧校准，不是完整60轨迹实验。十步固定输入采样执行通过；跨后端严格差异仍保留，LIBERO任务质量未评估。模型目录的runtime_contract.json说明已验证的外部诊断运行约定；原服务器并未完成验证。

官方仓库跟踪源码没有修改。新增文件保持相对工作区路径；upstream_reference仅为阅读附带的未修改算法源码。原大模型、完整Fisher、Python环境和构建目录不在本包中，不能据此单独从零重跑全部实验。

MANIFEST.json及SHA256SUMS.txt记录每个交付文件的哈希。历史文档保留先前阶段的结论，最新完成情况以QUANTIZED_MODEL_DEMO.md和completion.json为准。
""", encoding="utf-8")
    for name in ("SOURCE_STATUS.json", "README.md"):
        add(staging / name, name)

    manifest = []
    for name, path in sorted(files.items()):
        digest = sha256(path)
        if path == model_dir / "pi05.gguf" and digest != EXPECTED_MODEL_SHA:
            raise ValueError("Final model hash differs from validated artifact")
        manifest.append({"path": name, "bytes": path.stat().st_size, "sha256": digest})
    manifest_path = staging / "MANIFEST.json"
    manifest_path.write_text(json.dumps({"release": args.name, "files": manifest}, ensure_ascii=False, indent=2) + "\n")
    sums = staging / "SHA256SUMS.txt"
    sums.write_text("".join(f"{item['sha256']}  {item['path']}\n" for item in manifest))
    add(manifest_path, "MANIFEST.json")
    add(sums, "SHA256SUMS.txt")

    expected = {item["path"]: item["sha256"] for item in manifest}
    expected.update({"MANIFEST.json": sha256(manifest_path), "SHA256SUMS.txt": sha256(sums)})
    with archive.open("xb") as stream:
        with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for name, path in sorted(files.items()):
                tar.add(path, arcname=args.name + "/" + name, recursive=False)
    # Verify every archived byte while also hashing the archive for transport.
    with tarfile.open(archive, "r:") as tar:
        seen = set()
        for member in tar:
            name = member.name.removeprefix(args.name + "/")
            if not member.isfile() or name not in expected or name in seen:
                raise ValueError(f"Unexpected tar member: {member.name}")
            digest = hashlib.sha256()
            with tar.extractfile(member) as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected[name]:
                raise ValueError(f"Archive hash mismatch: {name}")
            seen.add(name)
        if seen != set(expected):
            raise ValueError("Archive is incomplete")
    digest = sha256(archive)
    checksum_path = archive.with_suffix(".tar.sha256")
    checksum_path.write_text(f"{digest}  {archive.name}\n")
    result = {"status": "PACKAGE_VERIFIED", "archive": str(archive), "bytes": archive.stat().st_size,
              "sha256": digest, "files": len(files), "model_sha256": EXPECTED_MODEL_SHA,
              "windows_destination": "D:\\actquant", "transfer_status": "NOT_TRANSFERRED"}
    (exports / (args.name + "-package.json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

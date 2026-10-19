#!/usr/bin/env python3
"""Prepare a reviewable workspace snapshot for the explicitly authorized GitHub repo."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = "JinYuan-Ye/actquant"
TAG = "pi05-full-60ep-20261019-v1"
PINS = {
    "ActQuant": ("https://github.com/arashakb/ActQuant.git", "b64791125070652fe6b554e244fe809c79ef5246"),
    "openpi": ("https://github.com/Physical-Intelligence/openpi.git", "215abfb217dbac7d5f1273282331b9b1866c0479"),
    "lerobot-v044": ("https://github.com/huggingface/lerobot.git", "8fff0fde7c79f23a93d845d1a50e985de01f8b8a"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(destination, final=False):
    destination = Path(destination).resolve()
    if destination == ROOT or not destination.is_relative_to(ROOT / "exports"):
        raise ValueError("Publication checkout must be isolated under exports/")
    if destination.exists() and not (destination / ".publication-workspace").is_file():
        raise FileExistsError("Refusing to use an unowned publication directory")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / ".publication-workspace").touch()

    def write(name, text):
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def copy(path, relative=None):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected a regular file: {path}")
        target = destination / (relative or path.relative_to(ROOT))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

    upstream_manifest = {}
    for name, (url, revision) in PINS.items():
        source = ROOT / "src" / name
        actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision or subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--name-only"], text=True):
            raise ValueError(f"Pinned upstream changed: {name}")
        upstream_manifest[name] = {"url": url, "revision": revision, "strategy": "vendored_tracked_snapshot" if name == "ActQuant" else "pinned_git_submodule"}
    # Include the original paper's full tracked source, preserving its directory layout.
    original = ROOT / "src/ActQuant"
    names = subprocess.check_output(["git", "-C", str(original), "ls-files", "-z"]).decode().split("\0")
    snapshot = {"repository": PINS["ActQuant"][0], "revision": PINS["ActQuant"][1], "files": {}}
    for name in filter(None, names):
        copy(original / name)
        snapshot["files"][name] = sha(original / name)
    write("src/ActQuant/.upstream-snapshot.json", json.dumps(snapshot, indent=2)+"\n")
    for directory, extensions in (("reproduction/pi05_lerobot", {".py", ".cpp", ".md"}),
                                  ("deployment", {".py", ".sh", ".ps1", ".md", ".txt", ".lock", ".in"})):
        for path in sorted((ROOT / directory).iterdir()):
            if path.is_file() and path.suffix in extensions:
                copy(path)
    # Transfer helpers are useful examples, but a private server address is not a public default.
    transfer = destination / "deployment/download_teaching_release.ps1"
    transfer.write_text(transfer.read_text().replace("root@61.220.157.12", "root@YOUR_SERVER").replace("[int]$Port = 41234", "[int]$Port = 22"))
    for name in ("openpi-server-freeze.txt", "openpi-libero-freeze.txt", "lerobot-reference-freeze.txt", "lerobot-reference-environment.json", "transformers-patch-manifest.json"):
        copy(ROOT / "deployment/evidence" / name)
    for path in sorted((ROOT / "reproduction/evidence").glob("*.json")):
        copy(path)
    for path in sorted((ROOT / "reproduction/results").rglob("*.json")):
        if not path.is_symlink() and path.stat().st_size < 2*1024**2:
            copy(path)
    copy(ROOT / "src/ActQuant/LICENSE", "LICENSE")
    write("UPSTREAM_SOURCES.json", json.dumps(upstream_manifest, indent=2)+"\n")
    write(".gitmodules", "".join(f'[submodule "src/{name}"]\n\tpath = src/{name}\n\turl = {PINS[name][0]}\n' for name in ("openpi", "lerobot-v044")))
    write(".gitignore", """envs/
.cache/
.local/
build/
exports/
data/
reproduction/data/
__pycache__/
*.pyc
*.gguf
*.safetensors
*.npy
*.npz
*.pt
*.part[0-9]*
.publication-workspace
.env
*.pem
""")
    write("NOTICE.md", """# Sources and adaptations

This workspace includes the original ActQuant tracked source under src/ActQuant, pinned and verified by its .upstream-snapshot.json. Its upstream license is preserved. The source remains unchanged; LeRobot input adaptation, diagnostics, reproducible calibration and publication tools live outside that directory. OpenPI and LeRobot are pinned dependencies listed in UPSTREAM_SOURCES.json.

The quantized artifact is derived from lerobot/pi05_libero_finetuned_v044 revision 7152f13e34c6c115d808aea0e5d08aca74eac14b. Tokenizer provenance is google/paligemma-3b-pt-224 revision 35e4f46485b4d07967e7e9935bc3786aad50687c. Consult the upstream model repositories for model terms; the source-code LICENSE alone does not describe model provenance.

Operational handoff history, credentials, environments, original weights and datasets are not part of this repository. Transfer-script server defaults are replaced with placeholders. Runtime and statistical limitations are documented with the model results.
""")
    pipeline_path = ROOT / "reproduction/results/pi05-full-01/pipeline.json"
    pipeline = json.loads(pipeline_path.read_text())
    if final and pipeline.get("status") != "PASS":
        raise ValueError("Cannot finalize publication before the full pipeline passes")
    state_text = "全量量化和文件/单观测配对执行检查已完成" if final else "量化所需代码和说明已提供，全量Fisher计算进行中，模型尚未完成"
    stats_text = ""
    if final:
        artifact = json.loads((Path(pipeline["model"]).parent.parent / "artifact-audit.json").read_text())
        paired = json.loads((pipeline_path.parent / "paired-validation.json").read_text())
        if not artifact.get("full_calibration") or paired["status"] != "PAIRED_EXECUTION_PASS":
            raise ValueError("Full artifact/paired validation gates did not pass")
        stats_text = f"\n模型文件 {artifact['quantized_gguf_bytes']:,} 字节，126个目标矩阵实际平均 {artifact['actual_block_bpw']:.9f} bit/权重。单观测反归一化7维动作RMSE为 {paired['sampled_actions_denormalized_7d']['rmse']:.8f}。这是误差测量，质量状态NOT_EVALUATED，尚无LIBERO成功率。\n"
        model_dir = Path(pipeline["model"]).parent
        for path in model_dir.iterdir():
            if path.suffix == ".json":
                copy(path, "models/pi05-full-60ep/"+path.name)
        write("models/pi05-full-60ep/RELEASE.json", json.dumps({"release": f"https://github.com/{REPO}/releases/tag/{TAG}",
              "tag": TAG, "model_sha256": pipeline["model_sha256"], "calibration_frames": pipeline["calibration_frames"]}, indent=2)+"\n")
    write("README.md", f"""# ActQuant π₀.₅ 全量校准复现

{state_text}。使用 README 指定的 LeRobot checkpoint，按原论文仓库10/10/10/30轨迹配置，使用60条轨迹的全部11,777帧重新计算HSIC、位宽分配和Fisher，量化126个LLM目标。视觉、embedding与action expert保持基础精度。
{stats_text}
完整模型完成并通过验证后提供于：[GitHub Release](https://github.com/{REPO}/releases/tag/{TAG})。附件按不超过1GiB分片，下载Release中的全部附件后，在本地执行：

```bash
python deployment/restore_release_model.py --directory /path/to/downloaded-assets
```

此命令按MODEL_SHA256.json逐项检查分片，拼接成pi05.gguf并核对完整SHA256；已有模型不会覆盖。参数应替换为下载附件的目录，Windows也可使用例如D:/actquant/model。

## 代码与说明

- 原论文仓库完整源码：[src/ActQuant](src/ActQuant)，上游跟踪文件保持原样。
- 新增适配与方法入口：[reproduction/pi05_lerobot](reproduction/pi05_lerobot)。
- 数据格式及算法：[方法导读](reproduction/pi05_lerobot/METHOD_WALKTHROUGH.md)、[HSIC](reproduction/pi05_lerobot/HSIC_DEMO.md)、[位宽分配](reproduction/pi05_lerobot/ALLOCATION_DEMO.md)、[Fisher](reproduction/pi05_lerobot/FISHER_DEMO.md)。
- 主要问题与解决：[问题汇总](deployment/TEACHING_RELEASE_ISSUES.md)、[兼容性核查](deployment/PI05_CHECKPOINT_COMPATIBILITY.md)。问题汇总保留了教学阶段历史；正式规模数据以本次报告为准。
- 正式运行：[FULL_CALIBRATION_RUN.md](reproduction/pi05_lerobot/FULL_CALIBRATION_RUN.md)，原始审计和结果在[reproduction/results/pi05-full-01](reproduction/results/pi05-full-01)。
- 依赖版本与构建：[deployment](deployment)、[UPSTREAM_SOURCES.json](UPSTREAM_SOURCES.json)。

克隆本仓库后初始化两个固定版本依赖：

```bash
git submodule update --init src/openpi src/lerobot-v044
```

这条命令获取已固定提交的OpenPI和LeRobot源码。后续构建按deployment文档执行；ActQuant源码已随仓库提供，安装脚本会核验其快照。模型原始权重和校准数据需要按文档单独准备，未提交到Git历史。

## 验证范围

跨后端严格数值差异仍保留NUMERICAL_MISMATCH和原容差。量化前后比较采用相同C++探针与输入/初始噪声。原高层服务器尚未完成输入适配，未进行LIBERO闭环任务质量验收。本仓库不是原论文官方仓库，也不宣称已复现论文成功率。四帧教学版文档明确标为历史教学实验，与本轮全量结果区分。
""")
    # The snapshot verifier itself is run against the concrete staged source.
    subprocess.run([sys.executable, str(destination / "deployment/verify_source_snapshot.py"), "--source-dir", str(destination / "src/ActQuant"), "--revision", PINS["ActQuant"][1]], check=True)
    sensitive = re.compile(rb"(?:ghp_|gho_|github_pat_|hf_)[A-Za-z0-9_]{20,}|-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----")
    for path in destination.rglob("*"):
        if path.is_file() and ".git" not in path.relative_to(destination).parts and path.suffix != ".zip":
            if sensitive.search(path.read_bytes()):
                raise ValueError(f"Possible credential in selected publication file: {path.relative_to(destination)}")
    if not (destination / ".git").exists():
        subprocess.run(["git", "init", "-b", "main", str(destination)], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "-C", str(destination), "remote", "add", "origin", f"https://github.com/{REPO}.git"], check=True)
    return {"status": "READY" if final else "PREPARED_WAITING_FOR_FULL_MODEL", "directory": str(destination),
            "upstream_snapshot_files": len(snapshot["files"]), "target_repo": REPO, "release_tag": TAG}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.final), indent=2))

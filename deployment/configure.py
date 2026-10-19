"""Apply only the official OpenPI Transformers replacement inside our venv."""
from pathlib import Path
import hashlib, json, shutil, sys
root = Path(__file__).resolve().parent.parent
assert Path(sys.prefix).resolve() == (root/'envs/openpi-server').resolve()
import transformers
assert transformers.__version__ == '4.53.2', transformers.__version__
target = Path(transformers.__file__).parent
source = root/'src/openpi/src/openpi/models_pytorch/transformers_replace'
backup = root/'deployment/evidence/transformers-original'
records = []
for src in sorted(source.rglob('*.py')):
    rel = src.relative_to(source)
    dst = target / rel
    orig = backup / rel
    if dst.exists() and not orig.exists():
        orig.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dst, orig)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # UV_LINK_MODE=copy prevents this upstream-required patch affecting uv cache.
    shutil.copy2(src, dst)
    records.append({'file':str(rel),'official_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
                    'installed_sha256':hashlib.sha256(dst.read_bytes()).hexdigest()})
(root/'deployment/evidence/transformers-patch-manifest.json').write_text(json.dumps(records,indent=2)+'\n')
print('Official Transformers patch applied; original files backed up.')

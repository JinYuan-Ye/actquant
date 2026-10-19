"""Record exact installed distributions, source commits and artifact hashes."""
from pathlib import Path
import datetime, hashlib, json, subprocess
from zoneinfo import ZoneInfo
root=Path(__file__).resolve().parent.parent
out=root/'deployment/evidence'
def run(*args):
    return subprocess.run(args,cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stdout
manifest={'recorded_at':datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'root':str(root),'repositories':{},'environments':{},'artifacts':{}}
for name,rel in {'ActQuant':'src/ActQuant','OpenPI':'src/openpi','LIBERO':'src/openpi/third_party/libero'}.items():
    repo=root/rel
    diff=run('git','-C',str(repo),'diff','HEAD','--stat')
    assert not diff,(name,diff)
    manifest['repositories'][name]={'commit':run('git','-C',str(repo),'rev-parse','HEAD').strip(),'tracked_changes':diff}
for env in ['openpi-server','openpi-libero','build-tools']:
    python=root/'envs'/env/'bin/python'
    freeze=run('uv','pip','freeze','--python',str(python))
    (out/f'{env}-freeze.txt').write_text(freeze)
    manifest['environments'][env]={'python':run(str(python),'--version').strip(),'distributions':json.loads(run('uv','pip','list','--python',str(python),'--format','json'))}
base=run('/usr/bin/python3','-m','pip','freeze')
(out/'preinstalled-pip-freeze-after.txt').write_text(base)
manifest['preinstalled_python_unchanged']=base==(out/'preinstalled-pip-freeze.txt').read_text()
assert manifest['preinstalled_python_unchanged'],'Preinstalled Python packages changed!'
for p in sorted((root/'build/pi05/bin').iterdir()):
    if p.is_file(): manifest['artifacts'][str(p.relative_to(root))]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
manifest['disk']=run('df','-h',str(root))
manifest['gpu']=run('nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu','--format=csv,noheader')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
refs=['src/ActQuant/README.md','src/ActQuant/tools/pi0.5/01_setup_env.sh','src/openpi/pyproject.toml','src/openpi/uv.lock','src/openpi/examples/libero/requirements.txt','src/openpi/third_party/libero/requirements.txt','deployment/client-constraints.txt']
(out/'reference-sha256.txt').write_text(''.join(f'{hashlib.sha256((root/f).read_bytes()).hexdigest()}  {f}\n' for f in refs))
print('Inventory saved:',out/'manifest.json')
print('Preinstalled Python unchanged: PASS; official tracked source unchanged: PASS')

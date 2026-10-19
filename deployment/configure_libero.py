from pathlib import Path
import json
root=Path(__file__).resolve().parent.parent
# Prevent LIBERO's interactive import-time request for dataset location.
config_dir = root/'.config/libero'
config_dir.mkdir(parents=True,exist_ok=True)
libero = root/'src/openpi/third_party/libero/libero/libero'
(root/'data').mkdir(exist_ok=True)
config = {'benchmark_root':str(libero),'bddl_files':str(libero/'bddl_files'),
          'init_states':str(libero/'init_files'),'assets':str(libero/'assets'),'datasets':str(root/'data')}
p = config_dir/'config.yaml'
if not p.exists():
    p.write_text(''.join(f'{k}: {json.dumps(v)}\n' for k,v in config.items()))
print('LIBERO config ready.')

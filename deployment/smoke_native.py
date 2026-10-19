"""Check official binaries and enumerate the native GPU backend, without weights."""
from pathlib import Path
import ctypes as c
import json, subprocess
root=Path(__file__).resolve().parent.parent
out=root/'deployment/evidence'
bin_dir=root/'build/pi05/bin'
for name,args,expected in [('pi05',['--help'],0),('llama-quantize',['--help'],1),('quantize-vision',[],1)]:
    result=subprocess.run([str(bin_dir/name),*args],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=30)
    (out/f'{name}-help.txt').write_text(result.stdout)
    assert result.returncode==expected,(name,result.returncode)
    assert ('usage' in result.stdout.lower() or 'OPTIONS:' in result.stdout),name
    print(name,'usage PASS; expected exit code',expected)
g=c.CDLL(str(bin_dir/'libggml.so'))
g.ggml_backend_load_all()
g.ggml_backend_dev_count.restype=c.c_size_t
g.ggml_backend_dev_get.argtypes=[c.c_size_t];g.ggml_backend_dev_get.restype=c.c_void_p
for f in ['ggml_backend_dev_name','ggml_backend_dev_description']:
    fn=getattr(g,f);fn.argtypes=[c.c_void_p];fn.restype=c.c_char_p
g.ggml_backend_dev_type.argtypes=[c.c_void_p];g.ggml_backend_dev_type.restype=c.c_int
devices=[]
for i in range(g.ggml_backend_dev_count()):
    dev=g.ggml_backend_dev_get(i)
    devices.append({'name':g.ggml_backend_dev_name(dev).decode(),'description':g.ggml_backend_dev_description(dev).decode(),'type':g.ggml_backend_dev_type(dev)})
assert any(d['type']==1 and 'L40S' in d['description'] for d in devices),devices
(out/'native-devices.json').write_text(json.dumps(devices,indent=2)+'\n')
print('Native GGML CUDA backend: PASS',devices)

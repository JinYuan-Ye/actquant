from pathlib import Path
import importlib.util, json, sys
import numpy as np
import torch
import transformers
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi_client import msgpack_numpy
root=Path(__file__).resolve().parent.parent
assert torch.cuda.is_available()
x=torch.arange(1024,dtype=torch.float32,device='cuda').reshape(32,32)/1024
assert torch.allclose((x@x.T).cpu(),x.cpu()@x.cpu().T,rtol=1e-4,atol=1e-4)
torch.cuda.synchronize()
config=Pi0Config(pi05=True,pytorch_compile_mode=None)
print('PyTorch',torch.__version__,'runtime CUDA',torch.version.cuda,'GPU',torch.cuda.get_device_name(0))
print('Transformers',transformers.__version__,'PI0Pytorch import and Pi0Config: PASS')
import pi05
print('pi05 binding:',pi05.__file__)
p=root/'src/ActQuant/tools/pi0.5/serve_policy.py'
spec=importlib.util.spec_from_file_location('actquant_serve',p)
server=importlib.util.module_from_spec(spec);spec.loader.exec_module(server)
payload={'actions':np.arange(21,dtype=np.float32).reshape(3,7)}
encoded=server.MsgpackNumpyPacker().pack(payload)
np.testing.assert_array_equal(msgpack_numpy.unpackb(encoded)['actions'],payload['actions'])
print('Official server -> OpenPI client NumPy message compatibility: PASS')
print('No model constructed or checkpoint loaded.')

sys.path.insert(0,str(root/"src/ActQuant/tools/pi0.5"))
from ggml_quant import GGMLQuantizer
GGMLQuantizer()
print("Official exporter GGML library discovery: PASS")

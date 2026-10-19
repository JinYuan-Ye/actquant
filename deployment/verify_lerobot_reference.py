"""Environment-only checks: imports and device discovery, never instantiate a model."""
import importlib.metadata as md
import json
from pathlib import Path
import torch
from transformers.models.siglip import check
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.pi05 import processor_pi05
root = Path(__file__).resolve().parents[1]
assert check.check_whether_transformers_replace_is_installed_correctly()
assert torch.cuda.is_available()
result = {"versions": {name:md.version(name) for name in
          ("lerobot", "torch", "torchvision", "torchcodec", "numpy", "scipy", "transformers", "safetensors", "sentencepiece")},
          "gpu":torch.cuda.get_device_name(0), "cuda_available":True, "official_transformers_patch":True,
          "model_instantiated":False, "model_forward_executed":False}
print(json.dumps(result, indent=2))
(root/"deployment/evidence/lerobot-reference-environment.json").write_text(json.dumps(result,indent=2)+"\n")

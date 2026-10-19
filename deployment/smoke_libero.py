import json, os
from pathlib import Path
import numpy as np
import torch, numba, mujoco
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import websocket_client_policy
print('Client torch',torch.__version__,'NumPy',np.__version__,'numba',numba.__version__,'mujoco',mujoco.__version__)
registry=benchmark.get_benchmark_dict()
counts={name:registry[name]().n_tasks for name in ('libero_spatial','libero_object','libero_goal','libero_10')}
assert all(n==10 for n in counts.values()),counts
print('LIBERO task counts:',counts)
suite=registry['libero_spatial']()
task=suite.get_task(0)
bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
assert bddl.is_file(),bddl
states=suite.get_task_init_states(0)
print('Packaged initial states:',len(states))
env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=64,camera_widths=64)
try:
 env.seed(7)
 obs=env.reset()
 assert obs['agentview_image'].shape==(64,64,3)
 from OpenGL import GL
 renderer=GL.glGetString(GL.GL_RENDERER)
 print('EGL GL_RENDERER:',renderer)
 print('LIBERO reset and 64x64 image render: PASS',obs['agentview_image'].shape)
finally:
 env.close()
print('No policy or rollout evaluated; no demonstration dataset downloaded.')

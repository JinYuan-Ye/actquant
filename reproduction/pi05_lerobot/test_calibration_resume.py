"""CPU regression: interrupted Fisher accumulation resumes exactly, including RNG."""
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import run_calibration as runner


class ResumeTest(unittest.TestCase):
    def test_exact_replay(self):
        class Adapter:
            def __init__(self, *args):
                pass

            def build(self, calib, indices, device):
                return int(indices[0]), None

        calls = [0]
        fail = [False]

        def step(model, targets, accumulators, obs, actions):
            calls[0] += 1
            if fail[0] and calls[0] == 4:
                raise RuntimeError("injected interruption after snapshot")
            value = torch.rand(2, 3) + float(np.random.rand()) + obs
            accumulators["blk.0.attn_q.weight"] += value.square()
            return float(value.mean())

        def write(values, path, count, source):
            Path(path).write_text(json.dumps({k: v.tolist() for k, v in values.items()}, sort_keys=True))

        fisher = SimpleNamespace(__file__=runner.__file__, get_target_params=lambda m: [],
                                 freeze_non_target=lambda *a: None,
                                 init_accumulators=lambda t: {"blk.0.attn_q.weight": torch.zeros(2, 3)},
                                 fisher_step=step, write_imatrix_gguf=write)
        hsic = SimpleNamespace(__file__=runner.__file__)
        calib = {"state": np.zeros((6, 8)), "meta": {"total_frames": 6}}
        replacements = dict(load_raw=lambda p: calib, official_modules=lambda: (fisher, hsic),
                            InputAdapter=Adapter, load_calibration_model=lambda p: (object(), None, []),
                            precision_snapshot=lambda: {})
        with TemporaryDirectory() as directory, patch.multiple(runner, **replacements), \
                patch.object(torch.cuda, "get_rng_state_all", return_value=[]), \
                patch.object(torch.cuda, "set_rng_state_all"), \
                patch.object(torch.cuda, "empty_cache"), \
                patch.object(torch.cuda, "max_memory_allocated", return_value=0), \
                contextlib.redirect_stdout(io.StringIO()):
            base = Path(directory)
            def invoke(name, resume=False):
                argv = [runner.__file__, "--stage", "fisher", "--calib-dir", str(base / "data"),
                        "--num-samples", "6", "--output", str(base / name), "--checkpoint-every", "2"]
                if resume:
                    argv.append("--resume")
                with patch("sys.argv", argv):
                    runner.main()
            invoke("continuous")
            calls[0] = 0
            fail[0] = True
            with self.assertRaisesRegex(RuntimeError, "injected interruption"):
                invoke("resumed")
            self.assertTrue((base / "resumed/fisher-resume.pt").is_file())
            fail[0] = False
            invoke("resumed", resume=True)
            for filename in ("fisher_llm.gguf", "fisher_vision.gguf", "losses.npy", "indices.npy"):
                self.assertEqual((base / "continuous" / filename).read_bytes(), (base / "resumed" / filename).read_bytes())
            self.assertEqual(json.loads((base / "resumed/run.json").read_text())["resumed_samples"], 2)
            self.assertFalse((base / "resumed/fisher-resume.pt").exists())


if __name__ == "__main__":
    unittest.main()

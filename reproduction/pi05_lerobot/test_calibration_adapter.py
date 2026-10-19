"""CPU regressions for the raw-input bridge; no model weights or experiments."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from calibration_adapter import ROOT, InputAdapter, load_raw


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = InputAdapter()
        cls.calib = load_raw(ROOT / "reproduction/data/pi05-readiness-01")

    def test_single_batch_axis_and_multi_sample_consistency(self):
        obs1, actions1 = self.adapter.build(self.calib, np.array([0]), "cpu")
        obs2, actions2 = self.adapter.build(self.calib, np.array([0, 1]), "cpu")
        for key in obs1.images:
            self.assertEqual(obs1.images[key].shape, (1, 3, 224, 224))
            torch.testing.assert_close(obs1.images[key], obs2.images[key][:1], rtol=0, atol=0)
        torch.testing.assert_close(actions1, actions2[:1], rtol=0, atol=0)
        self.assertEqual(actions1.shape, (1, 50, 32))
        self.assertFalse(actions1[:, :, 7:].any())

    def test_tokens_normalization_and_state_match_saved_official_reference(self):
        obs, actions = self.adapter.build(self.calib, np.array([0]), "cpu")
        with np.load(ROOT / "reproduction/results/pi05-real-forward-01/inputs_and_reference.npz") as reference:
            for value, key in ((obs.state, "observation.state"), (actions, "actions"),
                               (obs.tokenized_prompt, "observation.language.tokens"),
                               (obs.tokenized_prompt_mask, "observation.language.attention_mask")):
                np.testing.assert_array_equal(value.numpy(), reference[key])

    def test_invalid_index_and_dimension_contract_rejected(self):
        with self.assertRaises(IndexError):
            self.adapter.build(self.calib, np.array([-1]), "cpu")
        with self.assertRaises(ValueError):
            self.adapter.official_builder(self.calib, np.array([0]), None, None, "cpu", 10, 32, 32, 200)

    def test_old_pil224_dataset_rejected(self):
        with self.assertRaisesRegex(ValueError, "native RGB"):
            load_raw(ROOT / "reproduction/data/pi05-real-probe-01")

    def test_mismatched_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)
            for key in ("pixel_values", "state", "gt_actions"):
                np.save(path / (key+".npy"), self.calib[key][:1])
            (path/"prompts.json").write_text(json.dumps(self.calib["prompts"][:1]))
            meta = {**self.calib["meta"], "total_frames": 1, "camera_order": ["wrong"]}
            (path/"metadata.json").write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "camera order"):
                load_raw(path)


if __name__ == "__main__":
    unittest.main()

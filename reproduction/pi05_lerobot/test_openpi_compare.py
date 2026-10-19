"""CPU-only comparison guardrails; no full model allocation, checkpoint load or GPU."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from openpi_compare import (ROOT, SavedInputPI05, build_config, compare_losses,
                            fill_tied_aliases, observation_from_arrays, read_reference,
                            strip_policy_prefix)
from runtime_precision import configure_matmul_precision


class ComparisonTests(unittest.TestCase):
    def test_precision_override_applied_after_constructor_setting(self):
        original = torch.get_float32_matmul_precision()
        try:
            # Simulate the unconditional setting in the OpenPI constructor.
            torch.set_float32_matmul_precision("high")
            untouched = configure_matmul_precision("upstream")
            self.assertEqual(untouched["float32_matmul_precision"], "high")
            overridden = configure_matmul_precision("highest")
            self.assertEqual(overridden["float32_matmul_precision"], "highest")
            self.assertFalse(overridden["cuda_matmul_allow_tf32"])
            with self.assertRaises(ValueError):
                configure_matmul_precision("unknown")
        finally:
            torch.set_float32_matmul_precision(original)

    def test_config_maps_lerobot_horizon_without_default10(self):
        cfg = build_config(ROOT / "models/pi05_libero")
        self.assertEqual((cfg.action_horizon, cfg.action_dim, cfg.max_token_len), (50, 32, 200))
        self.assertTrue(cfg.pi05)
        self.assertIsNone(cfg.pytorch_compile_mode)

    def test_prefix_mapping_is_strict_and_does_not_discard_unknowns(self):
        self.assertEqual(strip_policy_prefix({"model.unknown.weight": 3}), {"unknown.weight": 3})
        with self.assertRaises(ValueError):
            strip_policy_prefix({"model.x": 1, "x": 2})

    def test_alias_fill_only_uses_shared_parameter_identity(self):
        model = torch.nn.Module()
        model.first = torch.nn.Linear(2, 2, bias=False)
        model.second = torch.nn.Linear(2, 2, bias=False)
        model.second.weight = model.first.weight
        model.unrelated = torch.nn.Linear(2, 2, bias=False)
        value = torch.ones(2, 2)
        state = {"first.weight": value}
        self.assertEqual(fill_tied_aliases(model, state), [["second.weight", "first.weight"]])
        self.assertIs(state["second.weight"], value)
        self.assertNotIn("unrelated.weight", state)
        with self.assertRaises(RuntimeError):
            model.load_state_dict(state, strict=True)

    def test_elementwise_comparison_catches_equal_mean_but_different_values(self):
        reference = np.ones((1, 50, 32), np.float32)
        target = reference.copy()
        target[0, 0, 0], target[0, 0, 1] = 2, 0
        self.assertEqual(target.mean(), reference.mean())
        result = compare_losses(target, reference, atol=1e-6, rtol=1e-5)
        self.assertEqual(result["status"], "MISMATCH")
        self.assertEqual(result["elements_outside_tolerance"], 2)
        self.assertEqual(compare_losses(reference, reference, atol=0, rtol=0)["status"], "EXACT_MATCH")
        with self.assertRaises(ValueError):
            compare_losses(np.full_like(target, np.nan), reference, atol=1e-6, rtol=1e-5)

    def test_saved_reference_replay_preserves_inputs_even_with_train_true(self):
        checkpoint = ROOT / "models/pi05_libero"
        arrays, _ = read_reference(ROOT / "reproduction/results/pi05-real-forward-01",
                                   checkpoint, build_config(checkpoint))
        obs, actions, noise, time = observation_from_arrays(arrays, "cpu")
        # Only nn.Module bookkeeping; skip the full model constructor entirely.
        model = SavedInputPI05.__new__(SavedInputPI05)
        torch.nn.Module.__init__(model)
        rng_before = torch.random.get_rng_state().clone()
        result = model._preprocess_observation(obs, train=True)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        for actual, expected in zip(result[0], obs.images.values()):
            self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actions, torch.from_numpy(arrays["actions"])))
        self.assertTrue(torch.equal(noise, torch.from_numpy(arrays["noise"])))
        self.assertTrue(torch.equal(time, torch.from_numpy(arrays["time"])))

    def test_modified_reference_mask_rejected(self):
        source = ROOT / "reproduction/results/pi05-real-forward-01"
        checkpoint = ROOT / "models/pi05_libero"
        arrays, report = read_reference(source, checkpoint, build_config(checkpoint))
        arrays["image_masks.right_wrist_0_rgb"] = np.array([True])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "report.json").write_text(json.dumps(report))
            np.savez(output / "inputs_and_reference.npz", **arrays)
            with self.assertRaisesRegex(ValueError, "camera mask"):
                read_reference(output, checkpoint, build_config(checkpoint))


if __name__ == "__main__":
    unittest.main()

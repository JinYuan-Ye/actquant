"""Small CPU checks for real-probe extraction; no download or model loading."""
import copy
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from prepare_real_probe import extract, load_exporter, select_sample


class RealProbeTests(unittest.TestCase):
    def setUp(self):
        self.exporter, _ = load_exporter()
        # Asymmetric image makes accidental camera swaps/flips detectable.
        base = np.zeros((224, 224, 3), dtype=np.uint8)
        base[:70, :90, 0] = 255
        wrist = np.full_like(base, 80)
        self.images = [base, wrist]
        encoded = []
        for pixels in self.images:
            buf = io.BytesIO()
            Image.fromarray(pixels).save(buf, format="PNG")
            encoded.append(buf.getvalue())
        self.states = np.arange(60 * 8, dtype=np.float32).reshape(60, 8)
        self.actions = np.arange(60 * 7, dtype=np.float32).reshape(60, 7)
        self.features = {
            "steps/observation/image": ("bytes", [encoded[0]] * 60),
            "steps/observation/wrist_image": ("bytes", [encoded[1]] * 60),
            "steps/language_instruction": ("bytes", [b"Pick up cup"] * 60),
            "steps/observation/state": ("floats", self.states.ravel()),
            "steps/action": ("floats", self.actions.ravel()),
        }

    def test_alignment_preserves_camera_order_and_raw_action_window(self):
        pixels, state, actions, task, count, _ = select_sample(self.features, 10, 50, self.exporter)
        np.testing.assert_array_equal(pixels[0, 0], self.images[0])
        np.testing.assert_array_equal(pixels[0, 1], self.images[1])
        self.assertFalse(pixels[0, 2].any())
        np.testing.assert_array_equal(state, self.states[10:11])
        np.testing.assert_array_equal(actions[0], self.actions[10:60])
        self.assertEqual((task, count), ("pick up cup", 60))

    def test_short_tail_rejected_instead_of_fabricating_actions(self):
        with self.assertRaisesRegex(ValueError, "actual consecutive actions"):
            select_sample(self.features, 11, 50, self.exporter)

    def test_native_keeps_original_pixels_and_same_state_actions(self):
        # A 256px pattern exposes the irreversible PIL resizing in the legacy path.
        raw = np.zeros((256, 256, 3), dtype=np.uint8)
        raw[::2, ::3, 0] = 255
        buf = io.BytesIO()
        Image.fromarray(raw).save(buf, format="PNG")
        features = copy.deepcopy(self.features)
        for key in ("steps/observation/image", "steps/observation/wrist_image"):
            features[key] = ("bytes", [buf.getvalue()] * 60)
        native = select_sample(features, 0, 50, self.exporter, "native")
        legacy = select_sample(features, 0, 50, self.exporter, "actquant")
        self.assertEqual(native[0].shape, (1, 3, 256, 256, 3))
        self.assertEqual(legacy[0].shape, (1, 3, 224, 224, 3))
        np.testing.assert_array_equal(native[0][0, 0], raw)
        for index in (1, 2):
            np.testing.assert_array_equal(native[index], legacy[index])
        self.assertEqual(native[3:], legacy[3:])

    def test_missing_camera_and_nonfinite_action_rejected(self):
        broken = copy.deepcopy(self.features)
        del broken["steps/observation/wrist_image"]
        with self.assertRaisesRegex(ValueError, "Missing or invalid"):
            select_sample(broken, 0, 50, self.exporter)
        broken = copy.deepcopy(self.features)
        broken["steps/action"][1][0] = np.nan
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            select_sample(broken, 0, 50, self.exporter)

    def test_existing_output_rejected_before_reading_data(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            marker = output / "existing.txt"
            marker.write_text("keep")
            with self.assertRaises(FileExistsError):
                extract(output / "missing_data", output, 0, 0)
            self.assertEqual(marker.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()

"""CPU checks for model transport integrity and pinned snapshot verification."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from publish_after_full_quantization import split_model
from restore_release_model import restore
from verify_source_snapshot import verify


class PublicationTest(unittest.TestCase):
    def test_split_and_restore_bytes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "original.gguf"
            payload = bytes(range(256))*10_000
            model.write_bytes(payload)
            assets = root / "assets"
            split_model(model, assets, hashlib.sha256(payload).hexdigest())
            self.assertEqual(restore(assets).read_bytes(), payload)
            with self.assertRaises(FileExistsError):
                restore(assets)

    def test_corrupted_part_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "original.gguf"
            model.write_bytes(b"verified payload")
            assets = root / "assets"
            split_model(model, assets, hashlib.sha256(model.read_bytes()).hexdigest())
            (assets / "pi05.gguf.part001").write_bytes(b"corruptd payload")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                restore(assets)
            self.assertFalse((assets / "pi05.gguf").exists())

    def test_multiple_parts_keep_exact_order(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            parts = [b"first\0"*100, bytes(range(256))*10, b"last\xff"*100]
            manifest = {"model_filename": "pi05.gguf", "model_bytes": sum(map(len, parts)),
                        "model_sha256": hashlib.sha256(b"".join(parts)).hexdigest(), "parts": []}
            for index, data in enumerate(parts, 1):
                name = f"pi05.gguf.part{index:03d}"
                (root / name).write_bytes(data)
                manifest["parts"].append({"filename": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            (root / "MODEL_SHA256.json").write_text(json.dumps(manifest))
            self.assertEqual(restore(root).read_bytes(), b"".join(parts))

    def test_snapshot_change_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "example.py"
            path.write_text("print('upstream')\n")
            (root / ".upstream-snapshot.json").write_text(json.dumps({"revision": "pinned", "files": {"example.py": hashlib.sha256(path.read_bytes()).hexdigest()}}))
            self.assertEqual(verify(root, "pinned"), 1)
            path.write_text("print('changed')\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                verify(root, "pinned")


if __name__ == "__main__":
    unittest.main()

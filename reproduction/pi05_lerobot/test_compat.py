"""CPU-only adapter checks; use real tokenizer if present. Never load weights."""
import copy
from pathlib import Path
import unittest
import torch
from compat import load_config, synthetic_sample, validate_preprocessing, load_strict, make_preprocessor, validate_tokenizer
ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "models/pi05_libero"


class StubTokenizer:
    def __call__(self, text, **kwargs):
        self.text = text
        shape = (len(text), kwargs["max_length"])
        return {"input_ids":torch.ones(shape,dtype=torch.int64),
                "attention_mask":torch.ones(shape,dtype=torch.int64)}


class CompatibilityTests(unittest.TestCase):
    def test_official_preprocessor_scales_and_state_prompt(self):
        from lerobot.policies.pi05 import processor_pi05
        from lerobot.processor import PolicyProcessorPipeline, batch_to_transition, transition_to_batch
        cfg = load_config(CHECKPOINT)
        stub = StubTokenizer()
        pre = PolicyProcessorPipeline.from_pretrained(
            str(CHECKPOINT), config_filename="policy_preprocessor.json",
            overrides={"device_processor":{"device":"cpu"},
                       "tokenizer_processor":{"tokenizer":stub}},
            to_transition=batch_to_transition, to_output=transition_to_batch,
        )
        raw = synthetic_sample(cfg)
        out = pre(copy.deepcopy(raw))
        result = validate_preprocessing(raw, out, CHECKPOINT, cfg)
        self.assertEqual(result["action_shape"], [1,50,7])
        self.assertEqual(stub.text, out["task"])
        raw2 = copy.deepcopy(raw)
        raw2["observation.state"][0,0] += 0.2
        out2 = pre(copy.deepcopy(raw2))
        self.assertNotEqual(out["task"], out2["task"])
        broken = copy.deepcopy(out)
        broken["action"] += 1
        with self.assertRaises(AssertionError):
            validate_preprocessing(raw, broken, CHECKPOINT, cfg)

    @unittest.skipUnless((ROOT/"models/paligemma_tokenizer/tokenizer_config.json").is_file(),
                         "Local tokenizer assets not present")
    def test_real_tokenizer_padding_matches_checkpoint(self):
        from transformers import AutoTokenizer
        tokenizer_dir = ROOT/"models/paligemma_tokenizer"
        cfg = load_config(CHECKPOINT)
        pre = make_preprocessor(CHECKPOINT, tokenizer_dir)
        out = pre(synthetic_sample(cfg))
        report = validate_tokenizer(out, CHECKPOINT, tokenizer_dir)
        self.assertEqual(report["padding_side"], "right")
        self.assertLess(report["valid_tokens"], cfg.tokenizer_max_length)
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
        wrong = tokenizer(out["task"], max_length=cfg.tokenizer_max_length,
                          padding="max_length", padding_side="left", truncation=True,
                          return_tensors="pt")
        broken = copy.deepcopy(out)
        broken["observation.language.tokens"] = wrong["input_ids"]
        broken["observation.language.attention_mask"] = wrong["attention_mask"].bool()
        with self.assertRaises(AssertionError):
            validate_tokenizer(broken, CHECKPOINT, tokenizer_dir)
        print("Real tokenizer regression:", report)

    def test_missing_weights_fail_before_model_construction(self):
        with self.assertRaises(FileNotFoundError):
            load_strict(ROOT/"reproduction/nonexistent_checkpoint")


if __name__ == "__main__":
    unittest.main()

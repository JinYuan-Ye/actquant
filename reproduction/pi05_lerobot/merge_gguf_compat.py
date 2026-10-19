#!/usr/bin/env python3
"""Call official pi05 merge with lossless GGUF metadata copying.

The pinned official copier reads parts[-1] for numeric arrays, which retains
only the final element with the installed reader. Tensor merge rules are reused.
"""
import argparse
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def copy_metadata(reader, writer, quant_type):
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        if key == "pi05.quant_llm" and quant_type:
            writer.add_string(key, quant_type)
        else:
            writer.add_key_value(key, field.contents(), field.types[0],
                                 field.types[1] if len(field.types) > 1 else None)


def merge(base, llm, output, quant_type=None, use_llm_embed=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    source = ROOT / "src/ActQuant/tools/pi0.5/merge_pi05_llm.py"
    spec = importlib.util.spec_from_file_location("official_pi05_merge_compat", source)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    official.copy_kv_metadata = copy_metadata
    official.merge(str(base), str(llm), str(output), quant_type, use_llm_embed=use_llm_embed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--llm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quant-type")
    parser.add_argument("--use-llm-embed", action="store_true")
    args = parser.parse_args()
    merge(args.base, args.llm, args.output, args.quant_type, args.use_llm_embed)


if __name__ == "__main__":
    main()

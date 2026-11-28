#!/usr/bin/env python3
"""Generate all-vocabulary FP32 CPU reference rows using pinned PyTorch/Transformers, without network."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform

from fetch_smollm2 import verify, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    manifest = verify(args.directory)
    os.environ.setdefault("HF_HOME", str(args.directory / ".reference-cache"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers", "safetensors", "numpy")}
    expected = {"torch": "2.6.0", "transformers": "4.46.3", "tokenizers": "0.20.3", "safetensors": "0.5.3", "numpy": "2.2.2"}
    if versions != expected:
        raise ValueError(f"Reference versions differ from the pinned requirements: {versions}")
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from transformers import LlamaForCausalLM
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    tokenizer = Tokenizer.from_file(str(args.directory / "tokenizer.json"))
    texts = ["The capital of France is", "Once upon a time", "Java is a",
             "café 中文 😀 123\n", "e\u0301 Ελληνικά\t42", "We're  here!\r\n2026",
             "abc12 Ⅳ½٣", "<|endoftext|>hello<|im_end|>", " \t\n", ""]
    vectors = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        vectors.append({"text": text, "ids": ids, "decoded": tokenizer.decode(ids, skip_special_tokens=True)})
    model = LlamaForCausalLM.from_pretrained(str(args.directory), local_files_only=True,
                                            torch_dtype=torch.float32, attn_implementation="eager").cpu().eval()
    cases = []
    logits_path = args.directory / "reference-logits.f32"
    row_number = 0
    with torch.inference_mode(), logits_path.open("wb") as output:
        for text in texts[:3]:
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            tokens = torch.tensor([ids], dtype=torch.long)
            result = model(tokens, use_cache=True)
            generated, rows = [], []
            for step in range(5):
                logits = result.logits[0, -1].float()
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite reference logits")
                values = logits.numpy().astype("<f4", copy=False)
                output.write(values.tobytes())
                top = torch.topk(logits, 5)
                rows.append({"row": row_number, "argmax": int(logits.argmax()),
                             "top_ids": top.indices.tolist(), "top_logits": top.values.tolist()})
                row_number += 1
                if step == 4:
                    break
                token = int(logits.argmax())
                generated.append(token)
                result = model(torch.tensor([[token]]), past_key_values=result.past_key_values, use_cache=True)
            cases.append({"text": text, "ids": ids, "generated_ids": generated,
                          "generated_text": tokenizer.decode(generated, skip_special_tokens=True), "rows": rows})
            print(f"Reference: {text!r} -> {cases[-1]['generated_text']!r}", flush=True)
    reference = {"schema": 1, "model": manifest["model"], "revision": manifest["revision"], "files": manifest["files"],
                 "reference": {"versions": versions, "python": platform.python_version(), "device": "cpu", "dtype": "float32",
                               "attention": "eager", "threads": 1, "logits_file": logits_path.name,
                               "logits_sha256": sha256(logits_path), "rows": row_number, "vocabulary_size": model.config.vocab_size},
                 "tolerance": {"absolute": 0.003, "relative": 0.0003}, "tokenization": vectors, "cases": cases}
    (args.directory / "reference.json").write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {row_number} complete vocabulary rows to {logits_path}")


if __name__ == "__main__":
    main()

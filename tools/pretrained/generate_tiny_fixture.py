#!/usr/bin/env python3
"""Regenerate small original BF16 Llama checkpoints and independent FP32 logits for portable CI."""
import json
import os
from pathlib import Path
import tempfile
from reference_environment import validate_versions

os.environ.setdefault("HF_HOME", str(Path(tempfile.gettempdir()) / "nnjava-reference-cache"))
os.environ["HF_HUB_OFFLINE"] = "1"
validate_versions()
import torch
import transformers
from safetensors.torch import save_file

torch.set_num_threads(1)
root = Path(__file__).resolve().parents[2] / "nnjava-integration-tests/src/test/resources/pretrained"
for tied in (True, False):
    torch.manual_seed(123)
    settings = dict(vocab_size=19, hidden_size=8, intermediate_size=12,
                    num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                    max_position_embeddings=32, rms_norm_eps=1e-5, rope_theta=100000,
                    tie_word_embeddings=tied, bos_token_id=0, eos_token_id=1)
    config = transformers.LlamaConfig(**settings)
    config.architectures = ["LlamaForCausalLM"]
    model = transformers.LlamaForCausalLM(config).to(torch.bfloat16).float().eval()
    model.set_attn_implementation("eager")
    directory = root / ("tied-llama" if tied else "untied-llama")
    directory.mkdir(parents=True, exist_ok=True)
    # Keep the tested checkpoint schema stable across reference-library serializer changes.
    portable = dict(settings, model_type="llama", architectures=["LlamaForCausalLM"],
                    hidden_act="silu", attention_bias=False, mlp_bias=False,
                    attention_dropout=0.0, pretraining_tp=1, transformers_version=transformers.__version__)
    (directory / "config.json").write_text(json.dumps(portable, indent=2) + "\n")
    weights = {name: tensor.to(torch.bfloat16).contiguous().clone() for name, tensor in model.state_dict().items()
               if not (tied and name == "lm_head.weight")}
    save_file(weights, str(directory / "model.safetensors"))
    with torch.inference_mode():
        tokens = torch.tensor([[2, 5, 3, 7]])
        logits = model(tokens, use_cache=False).logits[0].tolist()
    reference = {"seed": 123, "torch": torch.__version__, "transformers": transformers.__version__,
                 "weights": "Original seeded fixture, rounded to BF16; not pretrained model weights",
                 "input_ids": tokens[0].tolist(), "logits": logits}
    (directory / "reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    print(f"Wrote {directory}")

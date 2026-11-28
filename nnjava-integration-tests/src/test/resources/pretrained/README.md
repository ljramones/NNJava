# Original Llama fixtures

These tiny two-layer checkpoints are original seeded test data, not downloaded
pretrained weights. PyTorch 2.13.0 and Transformers 5.11.0 generate distinct tensors,
round weights to BF16, then compute FP32 eager-attention reference logits. One
fixture ties embeddings/readout; the other keeps them independent. Tests require
no Python packages, network access, model cache, or GPU.

Regenerate with the pinned environment in `tools/pretrained/reference-requirements.txt`:

```sh
python tools/pretrained/generate_tiny_fixture.py
```

Each `reference.json` records seed, versions, input IDs and all output logits.
The optional real SmolLM2 acceptance workflow is documented in
`docs/pretrained-model-validation.md`.

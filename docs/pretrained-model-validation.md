# Real pretrained-model validation

NNJava runs the published **HuggingFaceTB/SmolLM2-135M** checkpoint through the
Vectrix 4.1.0 scalar/SIMD CPU and native Metal exact/throughput backends on
DynamisGPU 1.14.1. Its 30 layers, grouped-query attention,
BF16 checkpoint storage, tied readout, tokenizer, KV cache and generation are
checked against an independent FP32 PyTorch/Transformers reference.

## Optional pretrained adapter

Add `org.dynamisengine:nnjava-pretrained` through the NNJava BOM and require
`org.dynamisengine.nn.pretrained` on the module path. This optional artifact depends
on IO/transformer/core/nn/formats. It has no numerical provider, tokenizer, training
or dataset dependency and is not included in the CPU convenience starter.

```java
var checkpoint = LlamaCheckpoint.open(Path.of("/tmp/nnjava-smollm2-validation"));
try (var context = ExecutionContext.cpu();
     var model = checkpoint.load(context, 64);
     var tokenizer = TokenizerAssets.huggingFace(
         checkpoint.directory().resolve("tokenizer.json"), TokenizerAssets.Options.gpt2())) {
    model.training(false);
    try (var generation = GenerationSession.builder().model(model)
            .sampling(s -> s.temperature(0)).maxNewTokens(4)
            .stopTokenIds(checkpoint.configuration().eosTokenIds().stream()
                .mapToInt(Integer::intValue).toArray()).build()) {
        var result = generation.generate(tokenizer.encode("Java is a"));
        System.out.println(tokenizer.decode(result.tokenIds()));
        // " programming language that is" for the pinned checkpoint.
    }
}
```

Imports: `org.dynamisengine.nn.pretrained.LlamaCheckpoint`,
`org.dynamisengine.nn.runtime.ExecutionContext`,
`org.dynamisengine.nn.tokenizers.TokenizerAssets`,
`org.dynamisengine.nn.generation.GenerationSession`, and `java.nio.file.Path`.
Select the Vectrix backend and tokenizer artifacts explicitly in the application.

`LlamaConfig` maps standard HF Llama configuration to the existing immutable causal
model definition. Supported geometry is full split-half RoPE, RMSNorm, bias-free
SwiGLU, MHA/MQA/GQA, and tied or separate embeddings/readout. Unsupported model
types, activation/bias switches, rotary scaling/partial/interleaved variants,
sliding windows, remote-code mappings and tensor partitions fail explicitly.
Family configuration validation alone does not establish another model's accuracy.

`LlamaCheckpoint` accepts exactly one local `model.safetensors` or
`model.safetensors.index.json`. It maps every expected HF tensor name and rejects
extra/missing tensors. A duplicate tied `lm_head.weight` must match embedding
shape, dtype and bytes before omission. BF16/F16 weights expand to FP32 for current
parameters; computation uses the selected numerical backend. Loading builds
explicit zero-initialized storage, so no provider normal initializer is required.
Loading closes an
unreturned model on failure. A backend execution fault remains recorded in its
borrowed context; discard a failed context according to its runtime contract.

The supplied cache capacity must fit the checkpoint configuration. Validation
uses 64 positions; the model's published maximum of 8,192 is not a tested
long-context acceptance result. Loading preserves the model's default training
mode, so callers explicitly select evaluation before cached generation.

## Reproduce the independent reference

The [manifest](../tools/pretrained/smollm2-135m.json) pins revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2` and SHA-256 hashes for the four assets.
The checkpoint is Apache-2.0, published by Hugging Face. Its 269 MB weight file and
reference environment stay outside Git. Java libraries perform local loading only.

```sh
python3 tools/pretrained/fetch_smollm2.py /tmp/nnjava-smollm2-validation
python3.12 -m venv /tmp/nnjava-reference
/tmp/nnjava-reference/bin/python -m pip install -r tools/pretrained/reference-requirements.txt
/tmp/nnjava-reference/bin/python tools/pretrained/generate_reference.py /tmp/nnjava-smollm2-validation
mvn -Pgpu,metal install                       # or mvn install for CPU-only consumption
mvn -f examples/pom.xml verify
python3 tools/verify_module_consumers.py \
  --pretrained-dir /tmp/nnjava-smollm2-validation \
  --pretrained-report /tmp/nnjava-smollm2-validation/scalar.json
python3 tools/verify_module_consumers.py --simd \
  --pretrained-dir /tmp/nnjava-smollm2-validation \
  --pretrained-report /tmp/nnjava-smollm2-validation/simd.json
```

`fetch_smollm2.py --verify-only DIRECTORY` checks existing files offline and refuses
hash mismatches. The reference generator uses pinned PyTorch 2.13.0, Transformers
5.11.0 and tokenizers 0.22.2, FP32 CPU eager attention, one thread and local files.
Both reference scripts validate their environment against the requirements file.
These versions replace earlier vulnerable pins; see the
[dependency remediation record](evidence/dependency-security-2026-11-27.json).
It writes `reference.json` and a hashed `reference-logits.f32` containing every
vocabulary logit in 15 rows: prompt prefill plus four accepted decode steps for
each of three prompts. The Java consumer verifies the pinned asset hashes and
reference data hash before model execution. A compact reference record is kept
in [evidence](evidence/dependency-reference-2026-11-27.json); regenerate the full binary
reference with the script above.

## Acceptance checks

- Ten tokenizer vectors cover English, Unicode, combining marks, whitespace,
  individual Unicode numbers, special tokens and empty text.
- All 737,280 reference logits are compared with a fixed `0.003 + 0.0003 * abs(ref)`
  tolerance; greedy token IDs and decoded continuations must agree exactly.
- Each accepted token grows the KV cache; final cached logits match full-sequence
  execution, reset reproduces prefill, and inference returns to its allocation baseline.
- Public generation reproduces the four reference IDs and incremental decoded text.
- Scalar/SIMD reactors cover independent tiny BF16 tied/untied checkpoint references,
  configuration rejection, missing/conflicting weights and failure cleanup.

The tokenizer importer now also accepts the declared `Digits(individual_digits=true)`
then `ByteLevel` sequence used by this model. Its 235-byte alphabet is preserved;
encoding fails if an input requires an absent byte. The default byte tokenizer
still has all 256 bytes. Unicode classification follows the JDK; these fixtures do
not establish universal compatibility with all native tokenizer Unicode tables.

## Results and remaining work

Both scalar and SIMD CPU runs pass. The largest absolute reference-logit difference
is **0.000165462**, with RMS error **0.0000225020** using the patched reference stack. The three continuations include
`Java is a` → ` programming language that is`.

Timing reports contain load, prefill, four accepted decode steps and request time,
plus parameter bytes and observed heap use. Each prompt has one measured request
after correctness passes. These short acceptance timings are not a controlled
throughput benchmark, peak-memory measurement or GPU performance result.

Native Metal exact and throughput modes now pass the same 737,280-logit reference,
cache and generation checks without the CPU backend on their module path. Maximum
absolute reference errors are 0.000165462494 and 0.000246047974 respectively; the
acceptance tolerances and exact token/text expectations are unchanged. The
[GPU model-inference guide](gpu-model-inference.md) gives commands, device-cache
behavior, mode-specific evidence and the remaining serving work. The default
generation API uses host sampling. The explicit [device-generation path](gpu-sampling.md)
now also matches all three continuations in both Metal modes, with guarded zero
logit downloads and direct device-token feedback.

This establishes CPU and native Metal pretrained-model acceptance. Quantized
placement, paged-cache orchestration, batching,
long-context behavior, corpus training and further model-family adapters remain
separate milestones. The [verification record](evidence/dependency-security-2026-11-27.json)
links the captured scalar/SIMD reports and test commands.

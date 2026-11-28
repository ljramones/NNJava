# Pretrained weights and tokenizer assets

NNJava separates container parsing, weight placement, and tokenizer configuration.
Applications choose the model definition and supply explicit checkpoint mappings.
The generic reader does not infer a model family from an architecture string.

## Modules and ownership

| Artifact / JPMS module | Responsibility | Dependencies |
| --- | --- | --- |
| `nnjava-formats` / `org.dynamisengine.nn.formats` | GGUF v3, safetensors, local shard indexes, immutable descriptors and metadata | Jackson Core 2.21.9; no NNJava core, model, or numerical module |
| `nnjava-io` / `org.dynamisengine.nn.io` | Explicit name/layout mappings and host snapshots | formats, core, nn; no concrete backend, training, data, or GPU runtime |
| `nnjava-tokenizers` / `org.dynamisengine.nn.tokenizers` | Byte BPE assets and tokenizer providers | formats; no model or numerical modules |

`Safetensors.open(path)`, `Safetensors.openIndex(indexPath)`, and `Gguf.open(path)`
return an owned `TensorFile`. Closing it unmaps every owned file. `data(name)` is a
borrowed, read-only `MemorySegment` whose lifetime ends with that file. Descriptor
shapes are immutable and row-major; GGUF's fastest-first dimensions are reversed.
Index descriptor offsets refer to the owning shard; obtain bytes through `data`.
Do not modify source files or close a source while another operation uses it.

```java
try (var file = Gguf.open(Path.of("model.gguf"))) {
    System.out.println(file.metadata().get("general.architecture"));
    file.tensors().forEach((name, descriptor) ->
        System.out.println(name + " " + descriptor.shape() + " " + descriptor.type()));
}
```

Both readers check shapes, overflow, byte counts, duplicates, overlap, and file
bounds. Safetensors additionally requires complete data coverage. Shard indexes
must name sibling files, with no escaping symlinks, missing tensors, or unindexed
tensor payloads. JSON/header limits are 16 MiB for safetensors/JSON assets and
64 MiB for GGUF; metadata arrays/counts have explicit limits. GGUF supports
little-endian version 3, ranks 1–4, and power-of-two alignment up to 1 MiB.

## Explicit weight loading

```java
try (var context = ExecutionContext.cpu();
     var model = Linear.builder().inputFeatures(3).outputFeatures(2).build(context);
     var file = Safetensors.open(Path.of("model.safetensors"))) {
    var loader = WeightLoader.builder()
        .transpose("projection.kernel", "weight") // checkpoint [3,2] -> Linear [2,3]
        .map("projection.bias", "bias")
        .expandFloatingPointToFp32(true)
        .maxTensorBytes(256L * 1024 * 1024)
        .build();
    var report = loader.validate(file, model); // no parameter writes
    loader.load(file, model);
}
```

Imports: `org.dynamisengine.nn.formats.*`, `org.dynamisengine.nn.io.WeightLoader`,
`org.dynamisengine.nn.nn.Linear`, `org.dynamisengine.nn.runtime.ExecutionContext`,
and `java.nio.file.Path`. The same loader works with GGUF and sharded safetensors.

Every unique target parameter requires one mapping. Tied names are recognized by
shared tensor identity, reported as aliases, and assigned once. Mapping both sides
of a tie fails. Sources must also be accounted for: use `ignoreSource(name)` for
an intentional omission or `requireAllSources(false)` for a selected subset.
Ignored tensors are reported but their contents are not compared with tied weights.
One source may map to one unique target. Matrix transpose is explicit; general
axis permutation and model-family transformations need additional adapters.

For supported definitions, `loader.build(file, definition, context)` constructs
backend-zero-initialized parameters, validates complete mappings and loads every
weight before returning an owned model. It closes an unreturned model on failure.
This path works without a normal initializer and supports CPU and Metal GPU
contexts. `LlamaCheckpoint.load` uses it automatically. Definition support and GPU
acceptance are described in [GPU model inference](gpu-model-inference.md).

Structural validation completes before the first assignment. Loading then stages
one tensor, assigns it, synchronizes, and clears its gradient. The default budget
limits each source and expanded payload to 256 MiB; temporary snapshot/backend
copies add a constant multiple of one tensor. There is no second full-model copy.
Loading borrows model/source, preserves training mode/counters, and rejects active
gradient recording. Backend/allocation failures during assignment can leave a
partially loaded model: discard that model. Loading is not a rollback transaction;
do not share the model with inference or training during import.

`WeightLoader.readTensor(file, name[, maxBytes])` returns an owned host
`TensorSnapshot` for F32, F16, BF16, I32, or BOOL, retaining storage bits. It survives
source closure and can be materialized in a context. Current model parameters are
FP32; opt-in F16/BF16-to-FP32 expansion unpacks IEEE storage for model loading.
No low-precision computation or training is implied.

## Quantized storage

GGUF exposes original Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K,
Q6_K, and Q8_K blocks with checked complete rows and byte sizes. Their descriptors
and data need no native runtime. Generic `WeightLoader` rejects quantized model
assignment. Backend placement/repacking and quantized execution remain numerical
adapter work; no NNJava dequantization kernel or implicit CPU fallback is added.
Other GGML types and safetensors dtypes fail explicitly.

## Tokenizer assets

```java
try (var tokenizer = TokenizerAssets.gpt2(
        Path.of("vocab.json"), Path.of("merges.txt"))) {
    int[] ids = tokenizer.encode("Hello 😀");
    System.out.println(tokenizer.decode(ids));
}
```

`TokenizerAssets` is in `org.dynamisengine.nn.tokenizers`. GPT-2 loading decodes its
reversible byte alphabet, checks contiguous IDs and causal ranked merges, and
uses the JDK Unicode GPT-2-style boundary regex. Defaults register
`<|endoftext|>` as BOS/EOS; `Options.noSpecialTokens()` or explicit `Options`
support other byte-BPE vocabularies. Special recognition and BOS/EOS insertion
still require `EncodeOptions`. The `gpt2-files` provider accepts exactly
`vocabulary` and `merges` paths through `TokenizerProviders.create`.

`TokenizerAssets.huggingFace(path, options)` accepts a checked `tokenizer.json`
subset: BPE with ByteLevel pre-tokenization/decoding, optionally preceded by
individual Digits, `add_prefix_space=false`,
`use_regex=true`, string or pair-array merges, and simple special added tokens.
It rejects normalization, dropout, unknown-token fallback, byte fallback, alternate
boundary rules, truncation/padding, non-special added tokens, whitespace-stripping
matches, and template post-processors. It does not read `tokenizer_config.json`,
chat templates, SentencePiece, or Llama tokenizer rules. Match vocabulary size,
token IDs, control policy, Unicode behavior, and model configuration in the consumer.
HF imports can declare a sparse byte alphabet; unavailable input bytes fail explicitly.

## Verification and next milestone

Tests cover malformed containers, mapped lifetimes, shard boundaries, storage
types, explicit transpose/expansion, complete validation before writes, failure
cleanup, tied decoder logits and cached generation, and supported tokenizer assets.
Standalone JPMS consumers exercise format-only parsing, model-only weight import,
and tokenizer-file loading without training or numerical tokenizer dependencies.
The initial format fixtures are synthetic. A subsequent optional Llama adapter and
[real SmolLM2 validation](pretrained-model-validation.md) now check a selected checkpoint
against reference tokens, all-vocabulary logits and complete CPU/Metal inference requests.
The [verification record](evidence/pretrained-loading-2026-11-27.json) records
243 passing scalar/SIMD/Metal reactor tests and the standalone consumer checks.
The later [GPU model-inference record](evidence/gpu-model-2026-11-27.json) covers
explicit supplied-weight construction, 300 passing reactor tests and real Metal
checkpoint acceptance.

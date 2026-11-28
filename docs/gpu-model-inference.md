# Supplied-weight GPU model inference

NNJava now constructs and loads dense networks and causal decoders on the optional
GPU backend. Complete **SmolLM2-135M** inference, device KV caches and public
generation pass on native Metal in both modes, targeting **Vectrix 4.1.0** and
**DynamisGPU 1.14.1**. The standalone consumer runs without the concrete CPU backend,
training or dataset modules on its module path.

## Construct initialized storage for supplied weights

```java
try (var model = definition.build(context, ParameterInitialization.ZEROS)) {
    StateArchive.read(weightsPath).restoreModel(model);
    model.training(false);
    // Use the ordinary model.forward(...) API.
}
```

`ZEROS` replaces normally random weight initialization with backend zero fill.
Biases and normalization scales retain their normal defaults. All parameters
remain trainable leaves; the policy does not expose uninitialized storage or
silently run a CPU random initializer. The existing `build(context)` keeps its
seeded behavior and still rejects a provider without normal initialization.

The policy is supported by Linear, dense sequential networks with activation,
dropout, normalization and layout layers, and complete causal decoders. Linear,
Embedding, attention, feed-forward and decoder-block constructors also accept the
policy explicitly. Custom definitions/layers must opt in; unsupported layer types,
including current convolution/recurrent loading constructors, reject it before
model allocation. Initialization policy is a construction choice, so serialized
architecture/configuration and seeded default behavior remain compatible.

For mapped checkpoints, `loader.build(file, definition, context)` constructs zero
storage, performs complete mapping validation, loads each unique parameter once
and returns the owned model. It closes an unreturned model on construction/load
failure; a recorded backend fault still requires discarding the failed context.
`LlamaCheckpoint.load` uses this path on CPU and GPU. Tied readouts retain their
embedding storage identity. The file can close after loading; model weights own
their device storage.

## Load and decode through the public API

```java
var checkpoint = LlamaCheckpoint.open(modelDirectory);
try (var context = ExecutionContext.open(providerId, deviceId);
     var model = checkpoint.load(context, 64)) {
    model.training(false);
    try (var session = model.newSession(1);
         var prefix = session.prefill(promptTokens);
         var next = session.decode(nextTokenIds)) {
        // Logits and per-layer cache tensors stay in this context.
    }
}
```

Select `vectrix-gpu-exact` or `dynamis-gpu-throughput` and a device from that
`BackendProvider.devices()` list, or construct `GpuBackend` with an owned
DynamisGPU device. Token tensors must belong to the selected context.

Sessions own fixed-capacity GPU K/V allocations. Appends use runtime copies into
strided slices; attention reads device views. Reset reuses storage, capacity and
batch checks precede mutation, and weight changes invalidate a populated session
until reset. Appends synchronize under the existing portable cache contract.
The existing `GenerationSession` path reads the final logit row and samples on the
host. The explicit [device-generation API](gpu-sampling.md) now samples on the GPU
and feeds sampled tensors directly into decode. Throughput mode also supports
the explicit [FP16 paged cache and recorded decode policy](gpu-paged-inference.md),
with separate whole-model precision acceptance. This guide's default path stays FP32.

One-token decoding exposed singleton matrix views with zero unused strides.
The GEMM adapter now canonicalizes those contiguous layouts before passing leading
dimensions upstream; no numerical copy is needed for that metadata correction.

## Verified reference and reproduction

The pinned SmolLM2 checkpoint has 30 distinct decoder blocks, GQA, a tied output
head, BF16 source weights expanded to FP32 and 538,060,032 bytes of unique FP32
parameter payload. Each mode checks ten tokenizer vectors, all 737,280 reference
logits, three four-token continuations, growing caches, full/cached parity, reset,
streaming text and allocation baselines. The reference is the independently
generated FP32 PyTorch/Transformers result; Python is reference tooling only.

| Native Metal mode | Maximum absolute reference-logit error | RMS error | Cache-check maximum error |
| --- | ---: | ---: | ---: |
| Exact | 0.000165462494 | 0.0000225020268 | 0 |
| Throughput | 0.000246047974 | 0.0000280550677 | 0.0000472068787 |

Both pass the unchanged acceptance bound `0.003 + 0.0003 * abs(reference)` and
exact generated token/text checks. For example, `Java is a` produces
` programming language that is`. These are measured results for this checkpoint
and short prompts, rather than a guarantee for other models or contexts.

After preparing the [pinned assets and independent reference](pretrained-model-validation.md):

```sh
mvn -Pgpu,metal install
mvn -f examples/pom.xml -Pgpu clean verify
python3 tools/verify_module_consumers.py --gpu-metal \
  --pretrained-provider vectrix-gpu-exact \
  --pretrained-dir /tmp/nnjava-smollm2-validation \
  --pretrained-report /tmp/nnjava-smollm2-validation/metal-exact.json
# Repeat with --pretrained-provider dynamis-gpu-throughput and a different report.
```

The runner explicitly resolves the selected Metal module, enables native access
and applies the published shader-resource JPMS opens. No CPU provider is included
in the GPU pretrained consumer's module path. Reactor tests also cover independent
tiny tied/untied BF16 references, biased MHA/LayerNorm/GELU decoders, dense networks,
stale caches, capacity rejection, singleton GEMM and construction/load failures.
See the [verification record](evidence/gpu-model-2026-11-27.json) and linked reports.

## Remaining work

Paged caches and recorded decoding now have an opt-in throughput policy. Next
measure controlled end-to-end performance and extend shared-cache orchestration.
Packing, synchronization and prefill scratch remain optimization targets. Reports contain acceptance timings,
not a throughput benchmark or GPU peak-memory measurement. Capacity 64 is verified
here; the checkpoint's published maximum context is not a long-context acceptance.

Quantized placement, mixed-precision execution/training, GPU normal initialization,
cross-entropy, stateful optimizers and convolution/pooling remain separate gates.
Vulkan execution and full GPU Trainer workflows remain unverified.

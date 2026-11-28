# Paged FP16 attention and recorded decoding

The optional GPU backend supports a single-sequence inference policy using
DynamisGPU **1.14.1** `PagedKvCache`, fused `AttentionF16` and `GpuRecording`,
alongside Vectrix **4.1.0**. Select `dynamis-gpu-throughput` explicitly. The default
session remains FP32; the exact provider rejects the FP16 policy before cache
allocation. No dependency versions or serialized model definitions change.

## Select the session policy

```java
var options = InferenceOptions.pagedFp16().withRecordedDecode();
model.training(false);
try (var session = model.newSession(1, options);
     var prefix = session.prefill(promptTokens);
     var next = session.decode(oneTokenTensor)) {
    var stats = session.replayStats(); // recordings, runs, current graph dispatches
}
```

Use `InferenceOptions.pagedFp16()` to run the same attention/cache path without
recording. For device sampling and direct tensor-token feedback:

```java
try (var generation = DeviceGenerationSession.builder()
        .model(model, InferenceOptions.pagedFp16().withRecordedDecode())
        .sampling(s -> s.temperature(0)).maxNewTokens(16).build()) {
    var result = generation.generate(promptIds);
}
```

Both `CausalModels.adapt` and `adaptDevice` accept the same options. The generation
module retains its provider-independent dependencies. Each request owns a fresh
cursor/cache; the model and execution context remain borrowed.

## Precision and capacity

Q/K/V are projected and rotary-transformed in FP32, then converted on the device
to FP16 with the published integer-exact, round-to-nearest-even `Cast` primitive.
K/V pages store those halves. Fused attention consumes FP16 Q/K/V and returns
FP32 output under `AttentionF16.fusedContract()`. Weights, residuals, normalization,
feed-forward operations and logits remain FP32. This policy changes numerical
results; it is separate from mixed-precision training.

Supported sessions have batch size one, an even head dimension through 256,
supported device group memory, and causal attention without an additional mask.
Ordinary paged sessions accept multi-token prefill and append. Recorded sessions
accept multi-token prefill followed by exactly one token per decode. Unsupported
policies, masks, batches and append shapes reject explicitly.

The default page size is 32 tokens. A custom `InferenceOptions` can select another
positive multiple of 32. Each session allocates its all-layer pool up front,
rounded to whole pages; logical page reservations grow with the sequence. Pool
K/V payload is `4 * layers * kvHeads * headDim * roundedCapacity` bytes, excluding
tables, live activations, scratch and recordings. This is not a measured peak-memory
figure or a shared serving pool. Reset reuses storage and discards the recording.
Weight mutation requires reset; failed compute requires a new session/context as
specified by the underlying execution error.

## What replay reuses

The first decode captures the public decoder's embedding lookup, all blocks,
normalization and readout into one upstream recording. Stable intermediate
bindings remain alive through the recording. Within a page, later calls skip the
Java model traversal, update rotary positions, patch each cache-write position,
update a device valid-length word and submit the recording. A page crossing closes
and rebuilds the graph against the new page table/key domain.

One-token attention uses the reserved page domain plus the live valid length,
so unused slots cannot influence its result. Prefill uses its actual key count
and causal offset. GPU-to-GPU input/output copies stay outside the recording;
the output copy preserves previously returned logits across later decodes.
Sampling remains a separate provider operation.

Recordings contain no readbacks. Each kernel-error check instead snapshots its
report at the original command position, then a separate ordered submission reads
those snapshots after execution. Length commits and callbacks follow successful
completion. Snapshots preserve errors even when a primitive reuses its report
buffer. Context/session closure and aborted captures release owned resources.

## Verification and reproduction

Native Metal tests cover FP32 agreement, ordinary/recorded paged parity, chunked
causality, growth through 70 tokens and two page boundaries, returned-logit
ownership, repeated requests, stale weights, capacity rejection, invalid device
tokens, debug kernels, injected capture failures and context-owned cleanup.
Instrumentation verifies that a repeated decode calls the ordinary backend only
for its two device copies, while the recorded graph supplies the entire decoder.
SmolLM2 passes 737,280 reference-logit checks per paged policy, with maximum
absolute error 0.012817740 and identical token/text continuations. Its complete
recording has 603 dispatches; a separate 31-token prefill plus four decodes checks
real-model page crossing and two graph builds. The original exact FP32 control
also passes its unchanged bound, with maximum error 0.000165462494.

After preparing the [pinned SmolLM2 assets](pretrained-model-validation.md):

```sh
mvn -Psimd,gpu,metal,benchmarks verify
mvn -Pgpu,metal -DskipTests install
mvn -f examples/pom.xml -Pgpu clean verify
python3 tools/verify_module_consumers.py --gpu-metal \
  --pretrained-provider dynamis-gpu-throughput \
  --pretrained-inference paged-fp16-replay \
  --pretrained-dir /tmp/nnjava-smollm2-validation \
  --pretrained-report /tmp/smollm2-paged-replay.json
# Repeat with --pretrained-inference paged-fp16 for the non-recorded control.
```

FP16 acceptance declares a separate whole-model gate of
`0.03 + 0.003 * abs(reference)` before running, plus identical greedy token/text
continuations. This is a checkpoint-specific engineering gate, not an upstream
kernel error bound. FP32 retains `0.003 + 0.0003 * abs(reference)`. The consumer
also guards device generation against FP32 logit downloads and scalar token
reuploads. See the [verification record](evidence/gpu-paged-inference-2026-11-27.json).

Controlled throughput and peak-memory benchmarks, long-context acceptance,
Vulkan verification, prefix sharing/fork/truncate, shared page pools, batching
and continuous scheduling remain separate work.

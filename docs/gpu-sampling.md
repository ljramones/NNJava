# GPU sampling and tensor-token generation

`DeviceGenerationSession` adds explicit provider sampling to the FP32 inference
path on **Vectrix 4.1.0 / DynamisGPU 1.14.1**. Logits remain GPU tensors and the
sampled INT32 allocation directly feeds the next embedding lookup. The application
reads one scalar token ID for streaming and stopping; runtime error reports and
metadata transfers remain separate from that four-byte token payload.
Greedy selection works in both modes. Stochastic device sampling requires
throughput mode; exact mode rejects it before dispatch.

## Generate with a device sampler

```java
model.training(false);
try (var generation = DeviceGenerationSession.builder()
        .model(model)
        .sampling(s -> s.temperature(.8f).topK(40).topP(.95f).seed(42))
        .maxNewTokens(16).minNewTokens(2)
        .stopTokenIds(eosTokenId)
        .build()) {
    var result = generation.generate(promptIds, token -> {
        output.accept(token.tokenId());
        return GenerationSession.Decision.CONTINUE;
    });
}
```

The prompt plus token budget must fit the model's context capacity. The session
borrows the model/context; each request owns a fresh cache and restarts its sampling
step at zero. Decode completes before callbacks, including the final budget/stop
token. Cursors, logits and temporary tensors close on normal completion, cancellation
or failure. The model remains in the caller's selected evaluation mode.

Stop IDs/sequences, minimum length, callbacks, cancellation and history-only
stopping criteria retain the existing token-generation conventions. Minimum-length
stop suppression uses a device selection mask. Only cancellation may come from
another thread. Callback reentry/closure and active gradient tapes are rejected.

## Choose the sampling contract explicitly

| API | Logits and feedback | Stochastic contract | Additional output/hooks |
| --- | --- | --- | --- |
| Existing `GenerationSession` | Host logits; selected IDs uploaded for decode | Vectrix/Philox host stream | Log probabilities, logits processors and repetition/presence/frequency penalties |
| `DeviceGenerationSession` | Tensor logits and direct sampled-token storage | Provider contract; current GPU sampler hashes seed/row/step | Token IDs and history-based control; no fabricated log probabilities |

Greedy selection uses Vectrix's GPU argmax twin, including first-tie, signed-zero
and subnormal behavior. Temperature/top-k/top-p sampling uses DynamisGPU's published
`Sampler` contract: repeatable on a device for the same seed and step, with device
exp differences allowed at draw boundaries. The draw stream differs from the host
Philox stream. Exact mode exposes only greedy selection, preserving its numerical
boundary. The existing host API is unchanged.

Device generation rejects non-default repetition/presence/frequency penalties;
there is no implicit host processing path. Custom tensor models can preprocess
logits with provider Ops before returning them. To integrate a different model,
implement `TensorAutoregressiveModel`; its cursor borrows INT32 token tensors and
returns owned FP32 `[vocabulary]` logits. `CausalModels.adaptDevice(model)` supplies
the standard decoder adapter.

## Primitive API, limits and errors

```java
var options = new SamplingOptions(.8f, 40, .95f, 42);
if (context.supportsSampling(options, rows, vocabulary)) {
    try (var tokens = Ops.sample(logits, options, step)) {
        // FP32 [..., vocabulary] -> INT32 [...], with no logit readback.
    }
}
```

`SamplingOptions` is in core's `org.dynamisengine.nn.ops`. Sampling is an optional
backend capability and never falls back to host sampling. The generation module
gains no native/GPU dependency; applications supply the optional GPU adapter.

Current GPU limits are 65,535 rows, a vocabulary through 1,048,576, top-k through
1,024 and unsigned 32-bit steps. Empty leading dimensions yield empty outputs.
Stochastic sampling needs 1,024-thread groups and a finite FP32 reciprocal
temperature. Greedy uses the 256-thread Vectrix twin. Stochastic top-p below one
requires an explicit top-k, avoiding the upstream primitive's implicit candidate
cap. Full-vocabulary sampling remains available with top-k zero and top-p one.

NaN, positive infinity and rows with every token banned fail through completion
with `KERNEL_ERROR`; individual negative infinities are valid bans. Validation
reduces/classifies rows on the GPU. Types, settings and dimensions are preflighted
before dispatch; sampling under a gradient tape is rejected.

## Verification and next integration

Native Metal tests cover strided/offset inputs, greedy ties/subnormals, full-width
seeds and steps, stochastic reference draws, a 70,001-token candidate/filter path,
empty rows, strict failures, minimum stops, repeat requests and abort cleanup.
Portable protocol tests reject any logit download and verify that sampled storage
is the actual decode input, including callbacks/cancellation/failure ownership.

The real SmolLM2 consumer validates all 737,280 reference logits and three
continuations in each Metal mode, then repeats generation with device sampling.
Its guarded backend rejects FP32 downloads during device generation, checks 12
scalar token reads and zero scalar token uploads, and verifies streamed text and
allocation baselines. Run the commands in the
[model-inference guide](gpu-model-inference.md); see the
[sampling verification record](evidence/gpu-sampling-2026-11-27.json).

Paged cache integration remains next. DynamisGPU's `PagedKvCache` stores FP16 and
feeds fused FP16 attention, requiring an explicit precision policy and new model
accuracy checks. Whole-model decode replay also remains planned: recordings
require stable bindings and cannot include readbacks/error checks. This batch
retains the verified fixed FP32 cache and normal per-operation recording. No
throughput speedup, Vulkan coverage or full GPU Trainer workflow is claimed.

# LLM Capabilities Roadmap

NNJava should support the full library workflow for constructing, loading,
executing, training, and adapting language models. Transformer primitives are
the foundation; usable LLM support also needs generation, token data, weight
loading, evaluation, and memory management.

This document describes the full target scope. The [main roadmap](roadmap.md)
distinguishes implemented and planned APIs. CPU configurable models, cache sessions,
single-sequence generation, and optional byte-level BPE now have implementations;
see the [generation guide](generation-guide.md). The main stack pins Vectrix 4.1.0
and DynamisGPU 1.14.1. General weight/tokenizer readers and the first optional Llama
adapter are implemented; SmolLM2-135M passes scalar/SIMD CPU and both native Metal
GPU modes against independent reference tokens/logits, with cached generation.
Further checkpoint compatibility, corpus workflows, adaptation,
shared-cache orchestration and scale features remain separate milestones. Explicit
[device sampling](gpu-sampling.md) is implemented with tensor logits/token feedback. The standalone
GGUF/Metal experiment retains its own dependency and accuracy limits.

The [paged inference policy](gpu-paged-inference.md) implements single-sequence
FP16 cache/attention and whole-decoder replay in throughput mode, with separate
SmolLM2 accuracy acceptance. Shared pools, prefix sharing and batching remain planned.

## Library and application responsibilities

The library supplies configurable architectures, token-based generation,
optional tokenizer implementations, generic weight readers, training objectives,
and reusable evaluation tools. Applications choose model configurations, weights,
vocabularies, corpora, prompt/chat formats, recipes, and deployment behavior.

Keep these capabilities independently consumable. Loading a pretrained model
must not require training modules. Generating token IDs must not require a
tokenizer implementation. Text applications can supply their own tokenizer or
choose an optional library provider. A model-family weight-name mapping is an
explicit adapter; opening a container does not establish architecture compatibility.

## Capability groups

### Model construction and architecture

- **LM-01 / LM-02:** fluent definitions for decoder stacks and their components:
  MHA, MQA and GQA; head dimensions and biases; normalization placement and epsilon;
  residual paths; GELU variants and gated feed-forward blocks; tied or separate
  embedding/readout weights. Existing primitives cover part of this foundation.
- **LM-06:** explicit position IDs, learned positions, rotary layout/base/scaling,
  causal/padding masks, sliding windows, and context-length validation. Each
  position/scaling scheme needs a named, versioned configuration and tested math.
- **LM-04:** planned encoder and encoder-decoder definitions, cross-attention, and
  reusable pooling/readout heads for embedding and sequence-to-sequence workloads.
- **LM-15:** later mixture-of-experts blocks: routing, top-k selection, expert
  composition, capacity policy, and auxiliary balancing losses. Distributed expert
  execution requires the separate distributed design.

Definitions must expose every supported architectural choice, serialize it, and
validate shapes/capabilities before allocating weights. Avoid model-family names
that silently imply unsupported options.

### Generation and inference

- **LM-03 / LM-07:** explicit inference sessions, prefill/decode, greedy and seeded
  sampling, temperature/top-k/top-p, repetition controls, EOS/stop conditions,
  token streaming, cancellation, and token log-probabilities.
- **LM-08 / EXEC-05:** batched requests with independent lengths and positions,
  paged KV storage, cache reset/fork/truncate, prefix sharing with safe ownership,
  memory limits, and admission/backpressure. Continuous batching follows a correct
  single-request path. Command replay must refresh bindings as cache pages and
  active batch shapes change.
  Single-sequence paged FP16 storage and page-scoped decoder replay are implemented;
  the remaining serving features above keep their own acceptance gates.
- **LM-16:** later beam search, constrained-decoding hooks, and speculative decoding.
  Speculative acceptance must preserve the selected target sampling distribution;
  each advanced strategy needs its own correctness and performance evidence.

Reusable single-sequence generation orchestration now lives in `nnjava-generation`. It consumes
the public transformer interfaces without a reverse dependency or server framework.
HTTP endpoints, chat sessions, tools, retrieval, and application scheduling remain
consumer concerns. CPU generation has the same public contract as GPU generation;
unsupported device capabilities must be reported explicitly.

### Tokenization and language-model data

- **LM-09:** an optional `nnjava-tokenizers` module with encode/decode contracts,
  vocabulary/special-token metadata, incremental decoding, and provider registration.
  Add byte-level BPE first, then explicitly supported SentencePiece model variants;
  provider implementations require compatibility fixtures for their formats.
- **LM-10:** shifted next-token targets, token-budget batches, padding and loss
  masks, sequence packing with document-boundary attention, reproducible shuffling,
  resumable token streams, and train/validation splits supplied by the caller.
- Tokenizers must test Unicode, configured normalization, unknown/special tokens,
  and incremental text boundaries. A chat template is caller-supplied data or a
  formatting adapter; it is not hard-coded into the model or generation engine.

### Weights, formats, and quantized inference

- **IO-04 / IO-05:** GGUF and safetensors readers, including tensor descriptors,
  metadata and supported shard indexes. Validate ranges, shapes, types, and
  required tensors. Support bounded loading without holding a second full copy
  of large weights in memory.
- **LM-17:** explicit weight-name/layout mappings, embedding/readout aliases,
  selective loading, quantized-weight placement, and precision/kernel selection.
  Keep format parsing in backend-neutral `nnjava-formats` and model import in
  `nnjava-io`; a provider materializes its
  arrays or GPU matrices from those descriptors and blocks.
- Reuse upstream quantized formats and GEMM. Document the actual supported block
  types and device paths rather than claiming compatibility with every GGUF model.

Initial IO-04/IO-05 and LM-17 infrastructure is implemented: mapped GGUF v3 and
safetensors/shards, explicit checked name/transposed-layout/tied-weight mappings,
and per-tensor staging. LM-09 includes GPT-2 files and a checked ByteLevel JSON
subset. [Supported boundaries](pretrained-loading.md) are explicit; quantized
placement and additional model families remain pending. The optional pretrained
module now supplies a standard Llama adapter and [SmolLM2 CPU acceptance](pretrained-model-validation.md).

### Training and adaptation

- **LM-10 / LM-14:** full-parameter causal-LM pretraining with token-normalized
  loss, microbatch gradient accumulation, schedules/clipping, activation
  recomputation, and exact training-state resume. Normalize accumulation by valid
  tokens across microbatches, not an unweighted mean of batch means.
- **LM-11:** supervised fine-tuning with arbitrary token loss masks, frozen
  subsets, validation, and resumable training. The application creates examples
  and decides which prompt/response tokens contribute to loss.
- **LM-05:** LoRA target selection, rank/scale/dropout, frozen base parameters,
  adapter-only persistence, and merge/unmerge where supported. Verify that base
  weights remain unchanged and only selected adapters receive gradients.
- **LM-12:** QLoRA after quantized-base input-gradient support and mixed-precision
  policies are verified. Quantized inference alone does not establish this support.
- **EXEC-03:** FP16/BF16 compute/storage choices, FP32 master weights and optimizer
  state, loss scaling, overflow handling, and resume. Capability checks apply to
  both forward and backward paths.
- **LM-18:** later preference objectives such as DPO, with paired-data contracts
  and reference-model handling. A reward-model/RL training platform needs a
  separate proposal.

Training remains an independent correctness and performance milestone. GPU
inference throughput does not predict training-step throughput or memory use.
Multi-device sharding and distributed training stay under DIST-01.

### Evaluation and diagnostics

- **LM-13:** token-weighted cross-entropy/perplexity, valid-token counts, validation
  sets, and plug-in downstream evaluators. Keep task datasets and scoring policy
  external; report tokenizer/context/truncation settings with every result.
- Compare loaded-model logits against a trusted reference, cached decode against
  full-sequence execution, and resumed training against uninterrupted execution.
- Measure prefill tokens/s, time to first token, decode tokens/s versus cache
  length, batch throughput, peak memory, allocation/transfer cost, and training
  step/token throughput. Separate cold initialization from warmed execution.
- Validate exact-mode compositions against their CPU reference, including the
  documented subnormal policy. Throughput checks use kernel accuracy contracts
  plus explicit end-to-end tolerances; per-kernel bounds do not automatically
  establish an entire model's error bound.

## Delivery order and acceptance

These are R5 submilestones. They can progress alongside classical NN work and
share R1/R2 foundations; training milestones reuse R3. GPU milestones also need
R6 provider integration and the relevant numerical coverage.

| Milestone                             | Scope                                                         | Observable completion                                                                                                                                                                                         |
|---------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| L1 Configurable models                | LM-01, LM-02, initial LM-06 and LM-13                         | Public builders construct a multi-layer decoder; definition round-trip, gradients, masks, tied weights, and cached/full logits pass.                                                                          |
| L2 Real-model inference               | LM-03, LM-07, initial LM-08/LM-09/LM-17, IO-04/IO-05, EXEC-05 | A selected pretrained decoder runs through an external consumer with distinct layer weights, output head, tokenizer, sampling, and growing KV cache; reference logits and full-run measurements are recorded. |
| L3 Pretraining                        | LM-10, initial LM-14, LM-13                                   | A small decoder learns a token corpus, reports held-out perplexity, accumulates variable token counts correctly, and resumes reproducibly; GPU training has separate measured evidence.                       |
| L4 Fine-tuning                        | LM-11, LM-05                                                  | Full and LoRA fine-tuning honor loss/freeze masks; saved adapters reload and reproduce logits, with merge parity where supported.                                                                             |
| L5 Scale and precision                | Remaining LM-06/LM-08/LM-14, LM-12, EXEC-03                   | Long-context, batching, recomputation, and precision policies pass targeted memory/correctness checks; QLoRA has independent backward validation.                                                             |
| L6 Architecture and objective breadth | LM-04, LM-15, LM-16, LM-18                                    | Encoder/decoder, MoE, advanced decoding, and preference objectives ship independently with their own acceptance tasks.                                                                                        |

L2 is the first GPU inference milestone after the shared model/provider foundation.
It does not wait for L3 training or every format/tokenizer provider. Choose one
compatible checkpoint, format, tokenizer, and device path first; expand the tested
matrix afterward. Small deterministic fixtures run in portable CI; large model
files are supplied explicitly for hardware/integration runs.

## Fluent API direction

Model, generation, tokenizer, loading, training, and adapter builders should compose.
Keep generation settings separate from architecture definitions and training state.
For example, this token-based CPU API is implemented:

```java
try (var session = GenerationSession.builder()
        .model(model)
        .sampling(s -> s.temperature(0.8f).topK(40).topP(0.95f).seed(42))
        .maxNewTokens(256)
        .stopTokenIds(eosTokenId)
        .build()) {
    session.generate(promptTokenIds, token -> {
        output.accept(token.tokenId());
        return GenerationSession.Decision.CONTINUE;
    });
}
```

The session borrows model weights; each request owns and closes a fresh cache.
The initial sampler uses host logits and Vectrix kernels/Philox, with a new
counter-based stream per request. `DeviceGenerationSession` now provides an explicit
provider-sampling alternative with its own draw contract and direct tensor-token feedback.
Text encoding/decoding wraps this token API without becoming a mandatory
dependency. Advanced callers retain direct forward/cache calls, custom loops,
and custom components.

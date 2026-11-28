# NNJava Feature Roadmap

NNJava is a Java 25 framework for creating neural networks and language models in
other applications. Its direction is a multi-module library with fluent builders,
reusable model definitions, and explicit execution/training lifecycles. LLM support
covers construction, loading, generation, pretraining, adaptation, and evaluation.
Applications choose their model, weights, vocabulary, data, recipes, and behavior;
the library supplies reusable mechanisms, including optional tokenizers.

This is a living proposal. Feature IDs stay stable as priorities change; new ideas
enter the backlog before they become implementation commitments.

## Current baseline and dependency targets

The modular CPU implementation supplies tensors, indexing, batched
GEMM, first-order autograd, neural functions, recursive modules, optimizers,
checkpoint/resume, data loading, and transformer primitives. The initial 54-test
CPU baseline has expanded through modular, builder, Trainer, and CNN milestones;
dated verification records live in `docs/evidence/`. Standalone consumers verify
the installed public APIs in addition to reactor tests.

The build pins **Vectrix 4.1.0** and **DynamisGPU 1.14.1**. Implemented now:
module/provider extraction, neutral archives, CPU/compatibility starters, BOM,
examples, fluent sequential/Linear construction, versioned model definitions,
configurable causal decoders, CPU prefill/decode sessions, supervised Trainer/data
APIs, grouped/depthwise CNN layers with pooling, and RNN/LSTM/GRU cells, masked
sequences, and explicit state. The optional GPU
foundation supplies storage, selected FP32 arithmetic, and rank-two GEMM; complete
GPU model execution remains pending.

The [model builder guide](model-builders.md) documents construction, definitions,
training-loop integration, cache lifecycle, and runnable fluent examples.

The earlier [design review](design-review-2026-11-25.md) explains the upstream
contracts. [Module migration](module-migration.md) documents implemented behavior,
experimental import changes, verification commands, and GPU capability limits.
The original 54 CPU tests pass on Vectrix 4.1.0; expanded tests cover isolation,
low-precision storage, legacy checkpoints, and real Metal execution. DynamisGPU
1.15 optimization work is outside the current pinned baseline and is not needed
for this implementation slice.

## Repository responsibilities

| Component               | Owns                                                                                                                                                                                                         |
|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| NNJava                  | Tensor/model semantics, differentiation, builders, layers, generation, optional tokenization, training/adaptation, evaluation, data contracts, weight formats, persistence, and numerical-provider adapters. |
| Vectrix and vectrix-gpu | CPU mathematics and kernels; exact (bit-identical) GPU twins of them.                                                                                                                                        |
| DynamisGPU              | The GPU runtime, and throughput GPU kernels (`dynamis-gpu-primitives`) with published accuracy contracts.                                                                                                    |
| Consuming application   | Its model/weight choices, tokenizer configuration and vocabulary, data, prompt/chat formats, recipes, explicit model-family mappings, and application behavior.                                              |

NNJava reuses both GPU kernel libraries:

- vectrix-gpu for an exact mode;
- DynamisGPU's primitives for a throughput mode.

New numerical requirements become upstream kernel requests (IDs G-E* and G-T* in
[numerical coverage](numerical-coverage.md)), never NNJava shaders.

## Approach and implementation order

| Phase                                  | Features                                                                                                                                                                                 | Depends on                                                                                                       | Completion gate                                                                                                                                                                |
|----------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| R0 Public design                       | Module boundaries, model-definition schema, builder conventions, ownership, extension points, and error reporting.                                                                       | Current baseline                                                                                                 | Reviewable NN, transformer, and training API examples with explicit lifecycle rules.                                                                                           |
| R1 Module extraction                   | Acyclic Maven modules, provider discovery, state/persistence separation, integration tests, a first-class examples module, and benchmark isolation.                                      | R0                                                                                                               | Existing behavior/tests and standalone consumer pass through packaged modules; CPU use has no GPU dependency.                                                                  |
| R2 Fluent NN creation                  | Layer builders, sequential definitions, deterministic naming/seeds, shape inference, initialization choices, and the MLP example.                                                        | R1                                                                                                               | Builder-created nonlinear networks match direct composition and produce actionable validation errors.                                                                          |
| R3 Training and persistence ergonomics | Trainer, data/optimizer/schedule builders, callbacks, evaluation, checkpoints, early stopping/cancellation, and autoencoder examples.                                                    | R2                                                                                                               | A normal NN can train, evaluate, save, reload, and resume through the public fluent API.                                                                                       |
| R4 Classical NN families               | CNN/pooling and recurrent primitives, RNN/LSTM/GRU state/sequence APIs, branch/residual builders, and runnable classical examples.                                                       | R2/R3                                                                                                            | CNN, RNN, LSTM, and GRU examples build/train through public APIs with derivative, state, and save/load checks.                                                                 |
| R5 LLM capabilities                    | Configurable transformers; loading/tokenization; generation and paged/batched inference; pretraining, SFT and LoRA; evaluation; later precision, architecture, and objective extensions. | R1/R2; R3 for training; R6 for GPU paths                                                                         | L1-L6 in the dedicated LLM roadmap: real-model inference first, then separately verified training and adaptation, with broader features delivered independently.               |
| R6 Published numerical stack           | EXEC-01 (Vectrix 4.1.0 CPU migration) first and independently; then GPU storage/completion, exact mode on vectrix-gpu, throughput mode on DynamisGPU primitives.                         | CPU migration verifies the published target; GPU adapter needs R1 provider SPI and confirmed operation coverage. | Supported exact-mode compositions match the CPU under the documented subnormal policy; throughput kernels meet their contracts and model-level checks; no hidden CPU fallback. |
| R7 Release and performance             | BOM/starters, compatibility policy, API guides, CI/platform matrix, and measured allocation/throughput improvements.                                                                     | Apply throughout; release after selected scope passes                                                            | A clean consumer resolves a coherent release and its documented examples/checkpoints work.                                                                                     |

R4 and R5 can progress independently once their
prerequisites are met. EXEC-01 comes before GPU dependencies because Vectrix 3.x
and 4.x cannot share a build. The 4.1.0 migration and R1 provider SPI are now
implemented; GPU operation coverage is the next integration gate. The first LLM GPU
milestone is a real loaded model with distinct weights, an output head, sampling,
and a growing cache. Training has its own correctness/performance milestone.
Neither LLM workstream waits for every classical architecture example. No delivery
dates are assigned until scope and dependencies are confirmed.

## Modules and planned extensions

The initial production boundaries are core, Vectrix CPU backend, NN/layers,
transformer/LM composition, data, model IO, and training. Add the optional GPU
backend (exact and throughput modes) once R1's provider SPI exists. A convenience starter and BOM
make consumption simple. Development modules cover integration tests, examples,
and benchmarks. `nnjava-examples` is a first-class reactor module, with the
[classical architecture catalog](examples.md) covering MLP, CNN, RNN, LSTM, GRU,
autoencoder, residual, and transformer examples.

`nnjava-generation` now supplies token-based inference orchestration and
`nnjava-tokenizers` supplies optional byte-level BPE/text conversion. Neither
requires training; generation accepts token IDs without a tokenizer dependency.
Both modules were introduced with working implementations and isolated consumers.

The [multi-module design](modules.md) defines responsibilities and dependency
arrows, including two necessary cuts: core must stop importing a concrete backend,
and optimizer snapshots/full checkpoints must stop creating training/IO cycles.
Keep tensor/Ops/autograd/runtime together initially because their current coupling
would make an immediate fine-grained split costly.

## Fluent API direction

Builders should make the common path short while retaining lower-level APIs:

- `NeuralNetwork.builder()` composes named layers and infers intermediate shapes.
- `CausalLanguageModel.builder()` composes configurable transformer components.
- Layer and optimizer builders expose advanced options through typed settings.
- `Trainer.builder()` orchestrates training without replacing explicit loops.
- Definitions can be built, inspected, saved, and reused without allocating model
  weights until a context is selected.

The [fluent builder proposal](fluent-builders.md) gives example syntax and
validation/ownership rules. NN and causal-model construction are implemented;
Trainer, optimizer definitions, loader builders, weighted evaluation, callbacks,
stopping, and loop checkpoints are implemented; see the [training guide](training-guide.md).
CNN and recurrent construction are documented in the [CNN guide](cnn-guide.md)
and [recurrent guide](recurrent-guide.md). Initial generation and byte-level BPE
are documented in the [generation guide](generation-guide.md). Branched builders,
additional tokenizer formats, and broader recurrent variants remain extensions.

## LLM capability track

The [LLM roadmap](llm-roadmap.md) makes language-model support a full feature track.
Its order is configurable models (L1), real-model inference (L2), pretraining (L3),
full/LoRA fine-tuning (L4), scale/precision (L5), and broader architectures/objectives (L6). Each milestone has an
observable acceptance task. The first inference run
needs one validated format/tokenizer/model path, not every planned provider.

The generic generation engine, tokenizer interfaces, format readers, adaptation
mechanisms, and evaluation tools belong in this library. Applications retain their
model choices, content, templates, recipes, and product behavior.

## Feature catalog

Statuses describe individual capabilities, so a working primitive does not imply
its planned builder or orchestration API already exists.

| ID                            | Feature                                                                                               | Current status                                                                                                    | Priority and phase    |
|-------------------------------|-------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|-----------------------|
| MOD-01                        | Module extraction and provider SPI                                                                    | Implemented with JPMS and class-path service registration                                                         | P0 R1                 |
| MOD-02                        | Neutral state contracts and model/training IO split                                                   | Implemented; legacy version-1 archive compatibility tested                                                        | P0 R1                 |
| API-01 | Immutable model definitions and configuration schemas | Implemented schema-1 configuration trees and checked definition files; migrations remain planned | P0 R0/R2 |
| API-02 | Layer and sequential NN builders | Implemented named sequential/Linear builders and built-in layer specifications | P0 R2 |
| API-03                        | Branched graph and residual/merge builders                                                            | Residual primitive exists; graph builder planned                                                                  | P1 R4                 |
| NN-01 | Dense/activation/embedding/dropout/norm layers | Dense/activation/dropout/norm specifications implemented; additional layer facades planned | P0 R2 |
| NN-02                         | Initialization, parameter freezing, and sharing APIs                                                  | Basic initialization/sharing exist; extensions planned                                                            | P1 R2/R4              |
| NN-03 | Convolution, pooling, and vision-oriented layers | CPU Conv2d (groups/depthwise, rectangular kernels, stride/padding/dilation) and max/average Pool2d implemented; broader vision layers remain extensions | P1 R4 |
| NN-04 | Vanilla RNN, LSTM, and GRU cells/layers with explicit state | Implemented CPU cells and sequence builders, ALL/LAST outputs, lengths, reset/detach, state snapshots, and full/chunked BPTT | P1 R4 |
| TRAIN-01 | Optimizers, groups, schedules, clipping, and losses | CPU primitives and immutable fluent AdamW/SGD settings implemented; advanced groups use factories | P0 R3 |
| TRAIN-02 | Trainer, callbacks, validation, early stopping, cancellation | Implemented supervised CPU orchestration for NN/LM modules with weighted evaluation and cooperative stopping | P1 R3 |
| DATA-01 | Deterministic batching and loader state | Implemented fluent loaders, configuration checks, and reproducible cursor/epoch restore | P0 R3 |
| DATA-02                       | Sequence padding/packing, length-aware batches, stream adapters                                       | Partial mask support; document-boundary packing and resumable token streams planned                               | P1 R3/R5 L3           |
| IO-01                         | Definition/model save/load and schema migration                                                       | Definition codec and model archives implemented; automatic schema migration remains planned                                                         | P0 R2/R3              |
| IO-02 | Full reproducible training resume | Trainer checkpoints include partial metrics, loop/stopping state and loaders; interrupted/resumed final archives match byte for byte | P0 R3 |
| IO-03                         | Extensible external tensor archive codecs                                                             | Planned codec extension points; explicit model-family mappings stay downstream                                    | P2 R5                 |
| IO-04 | GGUF container reader: neutral tensor/metadata/quantized-block descriptors | Implemented mapped v3 reader in independent formats module; quantized placement remains adapter work | P1 R5 L2 with EXEC-05 |
| IO-05 | Safetensors and supported sharded weight indexes; bounded loading | Implemented checked mapped files/sibling shard indexes and per-tensor staging | P1 R5 L2 |
| LM-01                         | Configurable attention, gated blocks, masks, norms, residuals, tied weights                           | CPU decoder blocks and configuration builders implemented; further variants planned                                                | P0 R5 L1              |
| LM-02 | Decoder and causal-language-model definitions/builders | Implemented CPU causal-model definition and builder with distinct block weights | P0 R5 L1 |
| LM-03 | Prefill/decode sessions and cache lifecycle | CPU prefill/decode sessions and stale-cache checks implemented; GPU session path pending | P0 R5 L2 |
| LM-04                         | Encoder/encoder-decoder, cross-attention, pooling/readout composition                                 | Planned architecture expansion                                                                                    | P2 R5 L6              |
| LM-05                         | LoRA selection/configuration, adapter save/load, merge/unmerge                                        | Planned; frozen-base and gradient checks required                                                                 | P1 R5 L4              |
| LM-06                         | Position schemes, rotary scaling, sliding windows, context validation                                 | RoPE/base and causal/explicit masks exposed; scaling/window policies planned                                                 | P1 R5 L1/L5           |
| LM-07 | Greedy/sampled generation, logits processors, stops, streaming, cancellation | Implemented single-sequence host sampling, penalties, token stops, streaming/cancellation, log probabilities and reusable schemas | P0 R5 L2 |
| LM-08                         | Batched/paged inference, prefix sharing, continuous batching, memory limits                           | Upstream paged cache exists; orchestration planned                                                                | P1 R5 L2/L5           |
| LM-09 | Tokenizer SPI, incremental decoding, optional BPE/SentencePiece providers | Implemented SPI/BPE/UTF-8 streaming, GPT-2 files and checked ByteLevel JSON subset; SentencePiece/further pipelines pending | P1 R5 L2 |
| LM-10                         | Causal-LM pretraining: shifted labels, token masks, packed document boundaries, token-normalized loss | Generic training primitives exist; LM workflow planned                                                            | P0 R5 L3              |
| LM-11                         | Supervised fine-tuning with token loss masks and selected trainable parameters                        | Planned; recipes/data remain caller-defined                                                                       | P1 R5 L4              |
| LM-12                         | QLoRA with frozen quantized base and trainable adapters                                               | Planned; quantized backward and precision prerequisites                                                           | P2 R5 L5              |
| LM-13 | Perplexity, token-weighted evaluation, reference-logit checks, LLM benchmarks | Whole-model gradient, cache parity and resume checks implemented; corpus evaluation/benchmark APIs planned | P0 R5 L1/L3 |
| LM-14                         | Token-correct gradient accumulation, activation recomputation, training memory policies               | Planned above existing explicit training/checkpoint APIs                                                          | P1 R5 L3/L5           |
| LM-15                         | MoE routing/expert blocks and auxiliary losses                                                        | Planned later expansion; distributed experts require DIST-01                                                      | P2 R5 L6              |
| LM-16                         | Beam search, constrained-decoding hooks, speculative decoding                                         | Planned later expansion; strategy-specific validation required                                                    | P2 R5 L6              |
| LM-17 | Weight-name/layout adapters, aliases, quantized placement and loading policies | Explicit names/matrix transposes, tied storage and bounded imports implemented; family adapters and quantized placement pending | P1 R5 L2 |
| LM-18                         | Preference objectives such as DPO and reference-model handling                                        | Candidate expansion; separate from a full RL platform                                                             | P2 R5 L6              |
| EX-01                         | First-class examples module and shared runner                                                         | Implemented with regression and tiny-transformer examples                                                         | P0 R1                 |
| EX-02 | MLP example | Implemented fluent XOR MLP with definition and weight save/load | P0 R2/R3 |
| EX-03 | Autoencoder example | Implemented bottleneck reconstruction with Trainer, shuffled data, evaluation, and exact resume | P1 R3/R4 |
| EX-04 | CNN example | Implemented two-convolution classifier with held-out evaluation, definition restore and exact training resume | P1 R4 |
| EX-05 / EX-06 / EX-07 | RNN, LSTM, and GRU examples | Implemented delayed-recall classifiers with held-out evaluation, definition restore and exact training resume | P1 R4 |
| EX-08                         | Residual CNN example                                                                                  | Planned follow-on                                                                                                 | P1 after CNN support  |
| EX-09 | Small causal transformer example | Implemented direct and fluent decoder examples with training, restore and cache checks | P0 R5 |
| EX-10                         | Encoder/decoder sequence example                                                                      | Planned later architecture expansion                                                                              | P2 R5 L6              |
| EX-11                         | Real pretrained-model inference consumer                                                              | Planned; reference logits, output head, sampling, growing KV cache, hardware measurements                         | P0 R5 L2              |
| EX-12 | Small LM pretraining and reproducible resume | Small token-ID training/resume implemented; corpus/packing workflow remains planned | P0 R5 L3 |
| EX-13                         | Supervised and LoRA fine-tuning examples                                                              | Planned; reload adapters and compare output                                                                       | P1 R5 L4              |
| EXEC-01                       | Vectrix CPU migration (`vectrix-core` and `vectrix-ml`)                                               | Implemented on 4.1.0; scalar/SIMD CPU checks pass                                                                 | P0 before R1/GPU      |
| EXEC-02                       | GPU adapter, exact mode (vectrix-gpu on the DynamisGPU runtime)                                       | Initial storage/arithmetic/GEMM subset implemented and tested on Metal; remaining NN operations pending           | P1 R6                 |
| EXEC-04                       | GPU adapter, throughput mode (`dynamis-gpu-primitives`)                                               | Initial storage/arithmetic/GEMM subset implemented on 1.14.1 and tested on Metal; remaining NN operations pending | P1 R6                 |
| EXEC-05                       | GPU inference path: `PagedKvCache`, GPU `Sampler`, recorded decode steps (`GpuRecording`)             | Upstream pieces exist                                                                                             | P1 R5/R6              |
| EXEC-03                       | Mixed precision and quantized execution policies                                                      | FP16/BF16 GEMM and quantized GEMM exist upstream; training support is request G-T9                                | P2 after R6           |
| PERF-01                       | Worker scheduling, memory planning, fusion, recomputation                                             | Initial benchmarks exist; optimizations planned                                                                   | P2 R7                 |
| DIST-01                       | Distributed training/execution                                                                        | Separate future design                                                                                            | P3                    |

P0 identifies foundations and minimum acceptance tasks for the selected milestone;
it does not make every P0 feature a prerequisite for the first modular release.
P1 completes normal workflows. P2 expands workloads and efficiency. P3 requires a
separately agreed design.

## Adding and prioritizing features

For each new request, record a stable ID, the user task it enables, owning module,
current primitive support, priority, prerequisites, and an observable acceptance
example. Identify upstream numerical requirements separately from NNJava API work.
If an addition changes persistence, ownership, or module dependencies, revisit R0
before assigning it to a phase. Reprioritization should preserve already-delivered
behavior and keep release gates explicit.

## First decisions and next design slice

The module/provider foundation, CPU migration, first NN/LM builders, definition
persistence, CPU cache sessions, supervised Trainer/data APIs, CPU CNN support,
RNN/LSTM/GRU cells/sequences/examples, token generation, and optional byte-level
BPE, neutral GGUF/safetensors readers and explicit pretrained mappings are implemented.
Next validate a selected real checkpoint and add model-family adapters,
branched/residual builders, expanded GPU numerical coverage, and
sequence-data packing/streaming. Preserve separate correctness and performance
milestones for real-model inference and LM training. Keep nonlinear classifiers
and caller-configured causal models as consumer API tests.

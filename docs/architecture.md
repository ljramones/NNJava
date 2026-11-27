# Neural Network Framework Architecture

This Java 25 framework supplies the reusable components needed to construct, train,
and execute neural networks. Applications assemble those components into their
own architectures, including LLMs, in separate repositories.

Status: architecture design with the CPU implementation delivered. The
[CPU guide](cpu-guide.md) and [roadmap](roadmap.md) identify implemented APIs and
remaining GPU/advanced work. The build baseline is JDK 25 and Vectrix 4.1.0. GPU support builds on
the published numerical stack as described in [GPU integration](gpu-integration.md)
and [numerical coverage](numerical-coverage.md). Current targets and implementation
status are recorded in the [roadmap](roadmap.md).

## Repository responsibilities

| Repository                             | Owns                                                                                                                                                                                                                                                                                |
|----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Vectrix (`vectrix-core`, `vectrix-ml`) | CPU numerical kernels: the reference semantics, with fixed-order, cross-platform deterministic results. Also the CPU side of quantization (`BlockQuant`, `Formats`).                                                                                                                |
| vectrix-gpu (Vectrix repository)       | Exact GPU twins of Vectrix kernels, bit-identical to the CPU (FP32).                                                                                                                                                                                                                |
| DynamisGPU                             | The GPU runtime (devices, pooled memory, arenas, queues, synchronization, recording/replay, pipelines, profiling) on Metal and Vulkan, and `dynamis-gpu-primitives`: throughput kernels (FP32/FP16/BF16/quantized) with published accuracy contracts, paged KV cache, GPU sampling. |
| NNJava                                 | Tensor/model semantics, differentiation, modules/builders, generation mechanisms, optional tokenizers, training/adaptation, evaluation, weight formats, persistence, and numerical adapters.                                                                                        |
| An application or LLM repository       | Architecture/weight choices, tokenizer configuration and vocabulary, datasets, prompt formats, training recipes, explicit model-family mappings, generation settings, and application interfaces.                                                                                   |

Reusable attention, positional encodings, masks, caches, and generic NN/LM model
builders belong in this framework. The application selects/customizes a model,
supplies its tokenizer and weights, and owns its recipe and behavior. The planned
`nnjava-examples` module demonstrates classical and transformer architectures
using the public library APIs.

The [LLM capability roadmap](llm-roadmap.md) covers model construction, real-model
generation, pretraining, SFT/LoRA, and later precision/architecture extensions.
Reusable generation algorithms and optional tokenizer implementations are library
capabilities; applications supply their settings and content.

## Dependency structure

Public framework APIs use framework types. Vectrix arrays, Vulkan handles, and
Metal objects stay inside adapters. Layer and training code call tensor operations;
the runtime resolves those operations to a backend implementation or a composition
of supported operations on the same device.

The current implementation is a multi-module Maven build with explicit
capability boundaries; see the
[module design](modules.md) for its dependency graph and required cycle cuts.
Keep numerical providers optional, model-only IO independent of training, and
benchmark/example dependencies outside production artifacts.

The [implemented builder API](model-builders.md) supplies immutable model definitions
and ergonomic construction above the existing components. Builders validate
shapes, configuration, sharing, and required capabilities before allocating
executable model state.

Use a provisional package root of `org.dynamisengine.nn`, divided as follows:

| Package                                 | Contents                                                                       |
|-----------------------------------------|--------------------------------------------------------------------------------|
| `tensor` and `ops`                      | Shapes, layouts, tensors, factories, and differentiable operations.            |
| `autograd`                              | Gradient recording, backward rules, and graph lifetime.                        |
| `nn`                                    | Parameters, modules, initialization, and common layers.                        |
| `transformer`                           | Attention components, positional encodings, masks, and KV caches.              |
| `generation` and `tokenizers` (planned) | Token-based decoding/session orchestration and optional text/token conversion. |
| `training` and `data`                   | Optimizers, schedules, batching, evaluation, and training helpers.             |
| `io`                                    | Named state, checkpoint metadata, and portable tensor persistence.             |
| `runtime` and `backend`                 | Execution contexts, capability discovery, and the backend SPI.                 |
| `backend.vectrix`                       | CPU operation mappings, scratch management, and scheduling.                    |

## Public abstractions

| Abstraction                     | Contract                                                                                                                                 |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `Tensor`                        | Typed values with a shape, strides, device, shared storage ownership, and optional gradient tracking.                                    |
| `Shape` and `TensorLayout`      | Immutable metadata with checked size arithmetic and explicit element offsets.                                                            |
| `Device` and `ExecutionContext` | Select an execution device and own its queue, allocation policy, worker resources, and random streams.                                   |
| `Ops`                           | Functional tensor operations with defined broadcasting, shape inference, numerical behavior, and backward rules.                         |
| `Parameter`                     | A named trainable leaf tensor with stable identity; tied weights share that identity.                                                    |
| `Module`                        | Registers parameters and child modules, exposes named state, and propagates training/evaluation mode.                                    |
| `UnaryModule`                   | The common single-input, single-output forward interface. Components such as attention use explicit input records for additional inputs. |
| `GradientTape`                  | Records differentiable operations and runs first-order reverse-mode differentiation.                                                     |
| `Optimizer`                     | Updates registered parameters and owns optimizer state and parameter-group settings.                                                     |
| `Dataset` and `DataLoader`      | Supply application-defined samples and batches with reproducible ordering.                                                               |
| `StateDict` and `Checkpoint`    | Persist named tensors, aliases, optimizer state, and training progress.                                                                  |

Expose an ordinary Java training loop first. An optional `Trainer` can manage
callbacks, evaluation, and checkpoint cadence without making custom loops depend
on it. Custom modules should be able to compose existing `Ops`; a custom primitive
also supplies shape inference, a backward rule, and backend implementations.

## Differentiation and parameter semantics

Use eager execution and a scoped, thread-confined gradient tape. Recording creates
one node per operation, rather than one node per tensor element. Reuse Vectrix's
matching backward kernels for fused operations where available.

Only floating-point tensors carry gradients. Scalar losses support `backward()`;
other outputs require an explicitly shaped upstream gradient. Backward accumulates
all paths to a shared tensor, reduces broadcast gradients, and sums repeated
embedding indices. Parameter gradients remain available until explicitly cleared.

Release saved activations after backward unless graph retention was requested.
Track storage mutation versions and reject writes that would invalidate a saved
value. Optimizer updates run after backward has released its saved values.
Training/evaluation mode controls layer behavior such as dropout; gradient
recording is controlled independently.

## Layer and training scope

The initial general-purpose layers are linear projection, embedding, ReLU, tanh,
sigmoid, GELU, SiLU, dropout, LayerNorm, and RMSNorm. Add convolution and pooling
after the tensor, differentiation, and training contracts are established.
The classical NN roadmap also includes vanilla RNN, LSTM, and GRU cells/sequence
layers. Their runnable architectures belong in `nnjava-examples`, as specified
by the [examples catalog](examples.md).

Provide mean-squared error and cross-entropy losses; SGD with momentum and AdamW;
parameter groups; gradient accumulation and clipping; warmup and decay schedules;
and evaluation hooks. Initialization and dropout use reproducible random streams.
Parameter traversal and gradient reduction have a documented stable order.

Checkpoints include tensor names, shapes, data types, aliases, optimizer slots,
schedule position, global step, and random stream counters. Reproducible resume
also requires application-provided sampler/data position and configuration. Save a
consistent snapshot after pending device work completes and before another update.

## Transformer building blocks

Provide components that applications can combine:

- Multi-head and grouped-query attention, including causal and padding masks.
- Embedding lookup, optional embedding/output weight sharing, and positional embeddings.
- Rotary position encoding with explicit position IDs, pairing layout, and frequencies.
- Normalization, residual composition, and gated feed-forward components such as SwiGLU.
- A session-owned KV cache supporting append, reset, valid lengths, and capacity checks.
  On the GPU it builds on DynamisGPU's `PagedKvCache` (pages, fork, truncate, pin),
  and a decode loop keeps tokens on the device through `Embedding` and the GPU
  `Sampler`, so logits need not return to the host.

Use `[batch, sequence, features]` at module boundaries and documented head layouts
inside attention. Sequence processing and cached single-token execution share
weights but have separate execution state. Training uses differentiable sequence
operations; mutable decoding caches operate outside gradient recording.

## Numerical and resource policy

Start with FP32 computation and gradients, INT32 indices, and boolean masks.
Put FP16 and BF16 in `DType` and the backend SPI from the start, as storage
types: the throughput GPU kernels take them as inputs (`GemmF16` takes FP32, FP16
or BF16; `AttentionF16` takes FP16), and adding a type later touches every
interface. Mixed-precision training (master weights, loss scaling) and quantized
execution remain later capabilities. FP64 can serve numerical reference checks
without requiring an FP64 neural-network backend.

The CPU adapter uses Vectrix's row-major array kernels, explicit scratch, and
fixed-order reductions. A GPU context runs in one of two modes:

- **Exact:** vectrix-gpu's twins; results equal the CPU bit for bit, apart from
  subnormals, which Apple GPUs flush.
- **Throughput:** DynamisGPU's primitives. Comparisons use each kernel's published
  `AccuracyContract` as the tolerance, rather than guessed ones.

No tier uses float atomics, so gradients never depend on GPU scheduling; NNJava's
own compositions keep that property.

Tensor handles support explicit release. Views and active operations retain their
underlying storage; tapes retain saved values. Each generation session owns its
working state while safely sharing read-only weights. See the
[backend contract](backend-contract.md) for lifetime and execution rules.

## Module and numerical integration decisions

The root is `nnjava-parent`; consumers use `nnjava`, capability modules, or the
`neuralnetworks` compatibility starter, all at `0.1.0-SNAPSHOT`. The package root
is `org.dynamisengine.nn`. A BOM and explicit JPMS descriptors are implemented.

The CPU backend uses `vectrix-core` and `vectrix-ml` 4.1.0 (EXEC-01, implemented).
The [module migration guide](module-migration.md) documents current GPU coverage.
The GPU adapter builds on vectrix-gpu, `dynamis-gpu-primitives` and the DynamisGPU
runtime. [GPU integration](gpu-integration.md) gives the storage, completion and
distribution details, and [numerical coverage](numerical-coverage.md) lists the
kernels requested upstream.

Native Metal and Vulkan platform checks, API/package migration, and stable release
policy remain explicit release gates. Decide the JPMS module names before the
module extraction (R1): Vectrix (`org.dynamisengine.vectrix`, `.ml`) and DynamisGPU (`org.dynamisengine.gpu.*`) are
named modules, and the GPU adapter needs Metal's
native access granted by module name.

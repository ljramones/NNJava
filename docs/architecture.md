# Neural Network Framework Architecture

This Java 25 framework supplies the reusable components needed to construct, train,
and execute neural networks. Applications assemble those components into their
own architectures, including LLMs, in separate repositories.

Status: architecture proposal. The library scope, JDK 25 baseline, Vectrix CPU
integration, and DynamisGPU integration are agreed requirements. The API names
below are proposed names.

## Repository responsibilities

| Repository | Owns |
| --- | --- |
| Vectrix | CPU numerical kernels, reproducible mathematical functions, and numerical kernel benchmarks. Target the 3.0 release line. |
| DynamisGPU | GPU discovery, buffers, transfers, queues, synchronization, shader pipelines, and dispatch through Metal and Vulkan. |
| NeuralNetworks | Tensor semantics, automatic differentiation, modules, training, parameter persistence, and backend adapters. The GPU adapter owns ML-specific kernels and their operation mappings. |
| An application or LLM repository | Model architecture, tokenizer, datasets, training recipes, pretrained model mappings, generation policy, and application interfaces. |

Reusable attention, positional encodings, masks, and cache storage belong in this
framework. A particular decoder architecture and its generation service belong
with the application. Small executable networks can demonstrate composition and
exercise integration tests.

## Dependency structure

Public framework APIs use framework types. Vectrix arrays, Vulkan handles, and
Metal objects stay inside adapters. Layer and training code call tensor operations;
the runtime resolves those operations to a backend implementation or a composition
of supported operations on the same device.

Begin with one primary Maven library containing the public API and Vectrix CPU
implementation. Add a separate, optional DynamisGPU adapter artifact when GPU
integration starts. Keep native dependencies out of the CPU deployment. Benchmarks
and examples can use separate build profiles or modules when introduced.

Use a provisional package root of `org.dynamisengine.nn`, divided as follows:

| Package | Contents |
| --- | --- |
| `tensor` and `ops` | Shapes, layouts, tensors, factories, and differentiable operations. |
| `autograd` | Gradient recording, backward rules, and graph lifetime. |
| `nn` | Parameters, modules, initialization, and common layers. |
| `transformer` | Attention components, positional encodings, masks, and KV caches. |
| `training` and `data` | Optimizers, schedules, batching, evaluation, and training helpers. |
| `io` | Named state, checkpoint metadata, and portable tensor persistence. |
| `runtime` and `backend` | Execution contexts, capability discovery, and the backend SPI. |
| `backend.vectrix` | CPU operation mappings, scratch management, and scheduling. |

## Public abstractions

| Abstraction | Contract |
| --- | --- |
| `Tensor` | Typed values with a shape, strides, device, shared storage ownership, and optional gradient tracking. |
| `Shape` and `TensorLayout` | Immutable metadata with checked size arithmetic and explicit element offsets. |
| `Device` and `ExecutionContext` | Select an execution device and own its queue, allocation policy, worker resources, and random streams. |
| `Ops` | Functional tensor operations with defined broadcasting, shape inference, numerical behavior, and backward rules. |
| `Parameter` | A named trainable leaf tensor with stable identity; tied weights share that identity. |
| `Module` | Registers parameters and child modules, exposes named state, and propagates training/evaluation mode. |
| `UnaryModule` | The common single-input, single-output forward interface. Components such as attention use explicit input records for additional inputs. |
| `GradientTape` | Records differentiable operations and runs first-order reverse-mode differentiation. |
| `Optimizer` | Updates registered parameters and owns optimizer state and parameter-group settings. |
| `Dataset` and `DataLoader` | Supply application-defined samples and batches with reproducible ordering. |
| `StateDict` and `Checkpoint` | Persist named tensors, aliases, optimizer state, and training progress. |

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

Use `[batch, sequence, features]` at module boundaries and documented head layouts
inside attention. Sequence processing and cached single-token execution share
weights but have separate execution state. Training uses differentiable sequence
operations; mutable decoding caches operate outside gradient recording.

## Numerical and resource policy

Start with FP32 computation and gradients, INT32 indices, and boolean masks.
Low-precision storage, mixed-precision training, and quantized execution are later
capabilities with their own accuracy contracts. FP64 can serve numerical reference
checks without requiring an FP64 neural-network backend.

The CPU adapter uses Vectrix's row-major array kernels, explicit scratch, and
fixed-order reductions. CPU/GPU comparisons use operation-specific numerical
tolerances; GPU execution does not inherit a claim of CPU bitwise equality.

Tensor handles support explicit release. Views and active operations retain their
underlying storage; tapes retain saved values. Each generation session owns its
working state while safely sharing read-only weights. See the
[backend contract](backend-contract.md) for lifetime and execution rules.

## Decisions to finalize during scaffolding

Choose the library's published name, artifact coordinates, and package root before
publishing APIs. Pin the final Vectrix 3.0.x release and verify its signatures during
adapter setup. Establish the first GPU hardware and OS test matrix, with native
Metal as the macOS integration target and Vulkan as another backend.

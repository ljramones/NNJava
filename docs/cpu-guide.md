# CPU Framework Guide

NNJava provides reusable tensor, differentiation, module, training, and transformer
components on JDK 25. Applications define architectures and data pipelines using
these components. The CPU implementation delegates numerical work to Vectrix.

## Build and consume the library

Run `mvn clean install` from the repository root. A separate project can depend on
`org.dynamisengine:nnjava:0.1.0-SNAPSHOT` (or the `neuralnetworks` compatibility starter). The independent project in
[`examples/consumer`](../examples/consumer/pom.xml) resolves that artifact and
assembles and trains a tiny transformer through public APIs:

```sh
mvn clean install
mvn -f examples/consumer/pom.xml compile exec:java
```

The numerical dependencies are `vectrix-core` and `vectrix-ml` 4.1.0. The root
is a multi-module aggregator; the [migration guide](module-migration.md) lists
capability artifacts, JPMS names, and experimental import changes. The `simd` profile selects Vectrix's SIMD classifier
and enables the
incubator Vector API in test JVMs. For an application running inside Maven's JVM, start Maven with
`MAVEN_OPTS=--add-modules=jdk.incubator.vector` when using the SIMD classifier.

The [NN/LM builder guide](model-builders.md) documents immutable definitions,
named sequential networks, configurable causal decoders, and cache sessions.
These use the same tensor/module/training contracts described below.

## Tensors and operations

`ExecutionContext.cpu()` owns a thread-confined CPU backend. Factories copy host
arrays into FP32, INT32, or BOOL tensors. FP16/BF16 raw `short[]` storage also
supports views, copies, and snapshots; arithmetic/gradients remain FP32. Set the floating-point factory's
`requiresGrad` argument to true to create a trainable leaf. Host reads return
copies and complete pending work.

`Shape` and `TensorLayout` use checked long metadata, while CPU allocations obey
Java array limits. Scalars have rank zero. Views retain shared storage;
`transpose`, `permute`, `slice`, `reshape`, and `broadcastTo` produce views when
valid. Reshape requires a compatible contiguous layout; call `Ops.contiguous`
first when materialization is needed. Broadcast views are read-only. Slices use
nonnegative bounds and positive steps.

Arithmetic broadcasts trailing dimensions. `Ops.matmul` accepts rank two and
higher, broadcasts batch axes, and multiplies the final two axes. Reductions
accept explicit axes and a `keep` flag. Sum over an empty domain is zero; mean,
maximum, and argmax reject empty reduction domains. Maximum/argmax select the
first logical occurrence on ties and the first NaN when present.

Gather follows integer-index selection along an axis, inserting the index tensor's
shape at that axis. Its backward sums repeated indices. Concatenation, stepped
slices, permutation, selection, and broadcast reductions also have backward rules.
Casts between different data types, comparisons, masks, argmax, and diagnostics
are not differentiated. Numeric casts use Java conversion behavior.

## Automatic differentiation and ownership

Use a scoped `GradientTape`, execute the forward graph inside its scope, and call
`tape.backward(loss)`. Non-scalar results require an explicitly shaped upstream
gradient. Backward is first-order and consumes the tape once. It accumulates all
paths to shared leaves, preserves accumulated leaf gradients between passes, and
releases saved graph values. `optimizer.zeroGrad()` clears leaf gradients before
the next independent step; omit it deliberately for gradient accumulation.

Close every returned tensor handle. `retain()`, `Parameter.value()`, and
`Tensor.gradient()` return additional owned handles. Views, tape records, and
pending backend work retain underlying allocations independently. Storage saved by
an active tape cannot be mutated. Optimizer updates and assignment run outside
recording. Closing a context drains work and invalidates its remaining tensors.

Training/evaluation mode is independent of recording. `model.training(false)`
disables dropout but does not disable a surrounding gradient tape. Without a tape,
operations execute without constructing a gradient graph.

## Network composition

`AbstractModule` supplies ordered recursive registration, parameter enumeration,
mode propagation, configuration metadata, and ownership. Custom modules implement
`UnaryModule` or expose explicit multi-input forward methods. Register children
and parameters with protected `register` methods. Override `localConfiguration`
for behavioral settings and `localState`/restore hooks for execution state.

Available components include `Linear`, `Sequential`, `Activation`, `Embedding`,
`Dropout`, `LayerNorm`, `RmsNorm`, and `Residual`. Linear applies to the last axis,
including sequence inputs. Its shared-weight constructor retains an existing
parameter value, allowing an output projection to share an embedding matrix.
Optimizers deduplicate shared parameter identities.

## Training and evaluation

`Losses` provides MSE and cross-entropy. Cross-entropy accepts `NONE`, `SUM`, and
`MEAN` reduction and an ignore index. Mean uses valid-target count; an all-ignored
batch produces zero loss and gradients. `Metrics.accuracy` follows the same
ignore-index convention.

`Sgd` supports momentum, Nesterov acceleration, and coupled decay. `AdamW` uses
decoupled decay and per-parameter moment counters. `ParameterGroup` supplies an
ordered name map, base learning rate, and decay; sharing a parameter across groups
is rejected. Built-in schedules include constant, step decay, and warmup/cosine.
Nonfinite gradients abort updates before parameters are changed. Use
`Gradients.clipGlobalNorm` between backward and the optimizer step when needed.

`Dataset` is the application extension point. `TensorDataset` batches named tensor
fields; `DataLoader` supplies deterministic ordering, optional shuffling/drop-last,
and a restorable epoch/cursor. A batch owns its tensor handles, and `Batch.get`
returns a retained owned handle. The loader borrows its dataset. Dataset contents
and identity remain application responsibilities.

## Persistence and exact resume

`io.StateArchive` provides model-only archives; `training.Checkpoint` adds optimizer
state. `TensorSnapshot` now lives in the core `state` package.

`StateDict.capture(model)` saves named values, tied-weight aliases, module
configuration, modes, and random counters. `Checkpoint.save(path, model,
optimizer, metadata)` also saves optimizer slots, counters, group configuration,
and schedule identity. Add `loader.state().metadata()` and application dataset or
recipe identifiers to the metadata map.

`Checkpoint.read(path).restore(model, optimizer)` validates names, shapes, types,
aliases, module behavior, and optimizer configuration before applying values.
Restore the loader separately using `DataLoader.State.fromMetadata`. Construct
compatible modules and optimizer groups first; initial weight seeds may differ
because saved values replace initialization. Dropout seeds/counters are restored.
For inference, call `training(false)` after restoring the model.

The format is versioned, streams big-endian primitive values and Java modified
UTF strings, and checks a CRC before exposing a loaded checkpoint. Ranks up to 64
are supported. Writes use temporary files and atomic replacement when available.
It is an NNJava state format; architecture-specific external weight formats belong
in downstream loaders. Tests verify subsequent CPU training steps match bit for
bit after resume under the same implementation and configuration.

## Transformer components

`Attention.scaledDotProduct` accepts Q/K/V shaped
`[batch, heads, sequence, headFeatures]`. It supports grouped-query sharing,
causal masks, padding masks, and optional probability dropout. Boolean masks use
true for allowed positions. Fully masked rows produce zero probabilities,
outputs, and score gradients.

`MultiHeadAttention` adds configurable projections, grouped-query heads, optional
rotary encoding, and attention dropout. Its module boundary is
`[batch, sequence, features]`. `RotaryEmbedding` supports offsets or explicit
position IDs, half/interleaved pairing, and a chosen frequency base. Explicit IDs
broadcast over axes preceding the feature axis, such as `[time]` or
`[batch,1,time]`. `GatedFeedForward` supplies a SwiGLU component. These components
can be combined with normalization and residual additions into application-owned
architectures.

For cached inference, set evaluation mode and give each attention layer/session
its own `KvCache`. `decode` supports prompt chunks and token-sized inputs, updates
positions from cache length, and enforces capacity. Cache mutation is prohibited
during gradient recording. Cached views are read-only and share mutable storage;
consume them before the next cache mutation.

## Numerical implementation and current limits

`VectrixBackend` calls BLAS/GEMM, Philox initialization, and optimizer kernels.
`CpuNeuralKernels` calls Vectrix activations, softmax, normalization, losses, and
RoPE. `CpuTensorKernels` implements layout traversal and indexing and uses Vectrix
reductions, norms, and reproducible transcendentals. Attention composes Vectrix
matrix products and softmax through differentiable tensor operations.

The CPU path remains single-context/thread-confined. There is no implicit device
fallback. A separate optional GPU adapter implements the [initial storage and
arithmetic subset](module-migration.md#initial-gpu-capabilities); full GPU model
execution remains pending.
Convolution/pooling and RNN/LSTM/GRU layers are planned in the
[feature roadmap](roadmap.md), with runnable architectures in the proposed
[examples module](examples.md). Nested tapes, higher-order gradients, graph
retention, mixed precision, and distributed execution remain later extensions.
API and checkpoint compatibility are experimental before a stable release.

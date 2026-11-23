# Tensor and Backend Contract

This proposed contract lets network code run through Vectrix on the CPU and
through DynamisGPU on supported GPUs. Tensor operations define mathematical
behavior; adapters implement that behavior with device-specific kernels.

## Tensor metadata and layout

- Each tensor has a data type, device, shape, element strides, storage offset, and
  reference to an allocation. Metadata uses checked `long` arithmetic.
- Scalars have rank zero. Zero-sized dimensions are valid. Sum over an empty
  domain returns zero; mean, maximum, and argmax reject an empty reduction domain.
- Contiguous storage is row-major. Broadcasting follows trailing dimensions:
  dimensions must match or one must equal one.
- Transpose and compatible reshape produce views. Reshape requires a compatible
  layout; `contiguous()` explicitly materializes a copy when necessary.
- Broadcast views use zero strides and are read-only. Initially support
  nonnegative strides; defer reversed views until their indexing contract exists.
- Initial data types are FP32, INT32, and BOOL. Arithmetic promotion is explicit;
  integer indices and masks do not acquire gradients.
- Operations require operands on the same device. Transfers are explicit. The
  initial device migration API operates between recorded training steps.

The Vectrix adapter validates each tensor against Java array indexing limits.
Long metadata supports device buffer descriptions and future storage providers;
it does not imply that the first CPU implementation supports arrays above those
limits. Public APIs provide typed host copies instead of exposing mutable backing
arrays.

## Backend interfaces

Keep this SPI separate from the user-facing tensor API:

| Interface | Responsibility |
| --- | --- |
| `BackendProvider` | Discover available devices and create backend contexts without initializing unused native backends. |
| `BackendCapabilities` | Report supported operation, data type, layout, size, and numerical-mode combinations. |
| `Buffer` | Own opaque device storage with byte size, alignment, device identity, and release semantics. |
| `KernelRegistry` | Resolve typed operation requests to direct kernels or registered same-device compositions. |
| `KernelCall` | Describe operation identity, input/output layouts, attributes, and overwrite/accumulate output semantics. |
| `ExecutionContext` | Own an ordered execution stream, scratch resources, allocation policy, and error state. |
| `Completion` | Report completion or failure and support explicit waiting. |

The framework infers shapes and validates structural arguments before execution.
The adapter receives validated layouts and preallocated outputs. Kernel support
includes backward requirements, rather than only forward availability. Preflight
a module's operation requirements before starting its training step.

Backend selection is explicit. An unsupported operation reports its operation,
device, data type, layout, and unmet capability. A registered composition may run
on that device; a CPU fallback requires an explicit application policy and visible
transfers.

## Required operation groups

The planned CPU coverage includes the following groups. The
[README](../README.md#implemented-first-slice) lists the currently implemented
subset. A GPU training backend must implement the corresponding operations or
same-device compositions for the features it advertises.

| Group | Required operations | Vectrix integration |
| --- | --- | --- |
| Storage and movement | Allocate, fill, copy, cast among supported types, contiguous copy, host upload/download. | Adapter-managed arrays, copying, and layout traversal. |
| Views and indexing | Reshape, transpose/permute, slice, broadcast, concatenate/split, gather, scatter-add. | Framework layout rules and adapter indexing kernels. |
| Elementwise math | Add, subtract, multiply, divide, negate, square, square root, exponential, logarithm, and scalar forms. | ReproducibleMath where applicable; adapter loops or reusable Vectrix extensions. |
| Conditions and masks | Equality/order comparisons, boolean and/or/not, and conditional selection with broadcasting. | Adapter kernels; selection gradients flow only through the selected value branch. |
| Reductions | Sum, mean, maximum, argmax, and broadcast-gradient reduction over selected axes. | Blasf reductions plus adapter axis traversal and explicit empty-input policies. |
| Linear algebra | Matrix multiplication with transpose options and batched matrix multiplication with broadcast batch dimensions. | Gemmf/GEMV per matrix; adapter batch iteration and per-worker scratch. |
| Neural functions | Activations, dropout, softmax/log-softmax, LayerNorm, RMSNorm, MSE, and cross-entropy. | Activations, Softmax, Norms, Loss; composed MSE and batching. |
| Embeddings and positions | Integer gather, repeated-index gradient accumulation, RoPE and its backward transform. | Adapter gather/scatter; Rope with explicit pairing and position conventions. |
| Training state | Zero/accumulate gradients, finite-value check, gradient norm/clipping, SGD and AdamW updates. | Blasf and Optim plus adapter state management. |
| Attention | Scaled dot-product attention, masks, grouped-query composition, and backward operations. | Attention per query/head with framework batch and sequence orchestration. |

Storage movement, shape-only transformations, random-number generation, indexing,
and elementary tensor operations require some adapter logic. Existing Vectrix
kernels remain the preferred numerical implementations. Propose new reusable CPU
math kernels upstream when a missing operation would otherwise be duplicated.

Fused normalization, attention, activation, and optimizer kernels are useful
optimizations. Their unfused semantics remain the reference contract. GPU
compositions must keep intermediate tensors on the device.

## Shape and numerical conventions

`Linear` stores weights as `[outputFeatures, inputFeatures]` and computes
`X * W^T + bias`. This matches a favorable Vectrix GEMM layout. Embeddings store
`[vocabularySize, features]`; repeated indices add into the same weight gradient.

Attention accepts an explicit scale and documents Q/K/V layouts. Query head count
must be divisible by KV head count. Causal masks and padding masks are combined
before softmax. A wholly masked row produces zero probabilities, zero output, and
zero score gradients; adapters must implement that rule around kernels that do
not supply it directly.

Losses expose `none`, `sum`, and `mean` reduction. For masked cross-entropy, mean
divides by the number of valid targets; an all-ignored batch returns zero loss
and zero gradients. Other nonfinite values propagate according to the operation's
documented numerical behavior; training can enable explicit finite-value checks.

Maximum and argmax break ties by the first logical row-major index in the reduced
domain. Maximum backward routes its gradient to that selected entry. Maximum
propagates NaN, and argmax selects the first NaN when one is present. Keep these
rules consistent across CPU reductions and GPU reduction trees.

Check dimensions, offsets, allocation ranges, and scratch requirements before
enqueueing. Data-dependent errors such as invalid embedding indices must surface
through completion without permitting out-of-range memory access. A failed step
must not proceed to an optimizer update.

## Execution and synchronization

CPU calls can complete synchronously; GPU calls can enqueue work. In both cases,
an execution context preserves operation order and inserts the dependencies needed
for its inputs and outputs. Host reads and checkpoint snapshots wait for relevant
work. An explicit synchronization method reports deferred execution errors.

Start with one ordered queue per context. Any later cross-queue sharing requires
explicit completion dependencies and resource visibility rules. Each concurrent
CPU task owns its scratch; GPU scratch stays live until submitted work finishes.

Use platform threads for sustained CPU compute. Divide Vectrix work into disjoint
output ranges and combine shared gradients in a stable order. Avoid nested
parallelism and oversubscription when a layer already schedules parallel work.

## Ownership and mutation

Closing a tensor invalidates that handle. Views, saved activations, and queued
operations retain the allocation independently. Native storage returns to a pool
only after its last owner releases it and its last use completes. Closing a
context drains pending work and reports failures before releasing its resources.
The context bounds the lifetime of tensors created through it; using them after
context closure fails explicitly. Tensor and context close operations are idempotent.

Outputs overwrite by default. Gradient accumulation and optimizer updates use
explicit mutation paths with declared read/write ranges. Reject unsafe aliasing,
including GEMM outputs overlapping inputs. Shared-storage mutation counters detect
changes to saved forward values before backward uses them.

Modules own their parameters; optimizers borrow those parameters and own their
state tensors. Temporary scopes own intermediate handles. Explicit release must
work for CPU arrays as well as native buffers, so moving to a GPU does not change
application lifetime rules.

## Randomness and numerical validation

CPU initialization, dropout, and sampling can use Vectrix Philox streams. Allocate
stable stream identifiers and counter ranges per logical operation; save counters
with checkpoints. GPU kernels use the same Philox counter mapping and uniform
conversion so dropout masks agree for the same logical inputs. Distribution
transforms can follow the floating-point tolerance contract. A GPU deterministic
mode also defines its reduction order and advertises the guarantee it implements.

Require numerical agreement between CPU and GPU using absolute and relative
tolerances selected for each operation, data type, and reduction size. Test
gradients with finite differences away from nondifferentiable points. Include
shared weights, broadcast views, repeated gathers, masks, empty inputs, and
overwrite/accumulate behavior. Kernel benchmarks report allocation and transfer
cost separately from execution time.

## Dependency integration

Target the Vectrix 3.0 release line and pin an exact released version when the build
is introduced. The operation mapping above follows the current public BLAS and
neural-kernel APIs; check it against the final 3.0 artifact.

Select either Vectrix's plain artifact or its `simd` classifier for a deployment.
The SIMD path needs `--add-modules jdk.incubator.vector`; keep a scalar validation
configuration too. The optional DynamisGPU adapter translates these operation
contracts to the generic compute services described in the
[GPU rework guidance](dynamisgpu-rework.md).

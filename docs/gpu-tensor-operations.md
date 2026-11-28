# GPU masks, indexing and rotary embeddings

The optional GPU adapter supports these operations on **Vectrix 4.1.0** and
**DynamisGPU 1.14.1**, in both exact and throughput modes. Native Metal checks cover
outputs, gradients, device layouts and cleanup. This extends the
[neural-operation batch](gpu-neural-operations.md); subsequent full-model results
are documented in [GPU model inference](gpu-model-inference.md).

## Implemented operations

| Operations | Implementation and contract |
| --- | --- |
| `CAST` among FP32, INT32 and BOOL | DynamisGPU `Cast`; Java saturation/rounding and nonzero-to-BOOL semantics |
| `EQUAL`, `LESS`, `GREATER` | `Logic`; exact FP32/int comparisons, including NaN, signed zero and large integers |
| `AND`, `OR`, `NOT`, `WHERE`, `IS_FINITE` | `Logic`; word selection excludes NaNs in an unselected branch |
| `ALL` | BOOL-to-FP32 conversion, minimum of zero/one flags, conversion back; the empty identity is true |
| `GATHER`, `SCATTER_ADD` | `Gather`; arbitrary axes and scalar/multidimensional indices; fixed serial accumulation per destination |
| `CONCAT`, `SLICE_BACKWARD` | Runtime buffer copies and zero fill; positive-step slice gradients |
| `REDUCE_MAX`, `ARGMAX`, `MAX_BACKWARD` | `Reduce.argmax`, indexed selection and gradient routing; first largest value or first NaN |
| `ROPE`, `ROPE_BACKWARD` | Vectrix `GpuRope`; exact CPU rotation in both modes, using INT64 arithmetic |

The adapter reports the selected tier and accuracy contract for each operation.
Word/index primitives are exact selections from DynamisGPU in both modes.
Scatter-add starts from positive zero and visits repeated indices in increasing
logical index order, matching NNJava's CPU accumulation. No float atomics are used.
Exact numerical comparisons retain the upstream subnormal/flush-to-zero limits.

Max returns the value at argmax's first winning index, preserving the sign of a
zero tie. It deliberately composes selection rather than using `Math.max`-style
reduction, whose zero-tie behavior differs. Multiple reduction axes preserve CPU
logical input order and restore gradient layouts.

## Layouts and device residency

FP32, INT32 and BOOL tensor programs support offsets, transpose/positive-stride
views, broadcasts and ranks above four through device packing. Gather flattens
the source into outer/axis/inner dimensions and flattens the index tensor, then
restores the public output shape. Empty indices produce empty gathers and zero
scatter gradients. Empty domains still validate any supplied indices.

Explicit RoPE positions are INT32 and broadcast to every axis preceding the
feature axis. They are validated and widened to INT64 on the GPU. Neither token
IDs nor explicit position IDs are read onto the CPU by these operations. Runtime
error checks transfer small status reports. Vectrix constructs inverse frequencies
on the host; implicit sequence positions come from the caller's long offset and
shape metadata. Rotation and its gradient run on the GPU, including offsets above
the INT32 range. Both half and interleaved feature pairing are supported.

Each call owns its temporary buffer handles; recorded commands retain their native
storage through submission completion or abort. The implementation adds no shaders
or CPU numerical fallback. Plans and packing are currently per call; caching,
fusion and dispatch reduction remain performance work.

## Validation and errors

Out-of-range gather/scatter indices and negative explicit positions fail through
completion with a runtime `KERNEL_ERROR`, wrapped by NNJava's context failure.
These are asynchronous data errors. Unsupported low-precision tensor computation
and missing RoPE INT64 capability are rejected before numerical dispatch. RoPE
uses Vectrix's even feature dimension and base greater than one requirements.
FP16/BF16 remain storage types for this adapter.

```sh
mvn -Pgpu,metal clean install
mvn -Psimd,gpu,metal,benchmarks verify
mvn -f examples/pom.xml -Pgpu clean verify
python3 tools/verify_module_consumers.py --gpu-metal
```

Hardware tests cover integer precision, NaN/signed-zero comparison and selection,
high-rank masks, all nine word-type cast pairs, empty identities and indices,
repeated-index cancellation, slice/concat gradients, max ties/NaNs and multi-axis
gradients. RoPE forward/backward matches CPU bits for both layouts, broadcast
positions through `Integer.MAX_VALUE`, scalar IDs and offsets above 2^40. Seven
successive rotary scratch allocation failures per mode check abort cleanup.

A supplied-weight embedding → RoPE → masked grouped-query attention composition
matches CPU outputs and embedding/value gradients; fully masked query rows are
zero. The standalone JPMS GPU consumer also composes device-produced token and
position IDs, causal attention, autograd and argmax without the CPU backend or
layer/transformer/training modules on its module path. See the
[verification record](evidence/gpu-tensor-2026-11-27.json).

## Remaining GPU milestones

The subsequent [GPU model-inference batch](gpu-model-inference.md) implements
supplied-weight construction/loading and complete transformer inference against
the independent pretrained reference in both Metal modes.
GPU normal initialization, cross-entropy, stateful optimizers, convolution/pooling
and mixed-precision training remain unsupported. [GPU sampling](gpu-sampling.md) is
now integrated; paged cache integration, recorded decode and quantized placement remain.

Existing host utilities such as `Masks.padding` still read lengths for validation
and construct masks on the host; this batch implements the underlying device Ops.
Vulkan execution, full GPU Trainer workflows and controlled model throughput remain unverified.

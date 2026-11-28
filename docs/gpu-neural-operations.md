# GPU neural operations

The optional `nnjava-backend-vectrix-gpu` adapter now supports a substantial FP32
neural-operation subset, using **Vectrix 4.1.0** and **DynamisGPU 1.14.1**. Both
exact and throughput modes are verified on native Metal. The implementation
composes published kernels and runtime copies; NNJava adds no shaders or implicit
CPU numerical fallback.

## Implemented coverage

| Operation | Exact mode | Throughput mode |
| --- | --- | --- |
| Add/subtract/multiply/divide, exp/log/sqrt | Vectrix twins; broadcasts/views packed on device | DynamisGPU strided operations through rank four; exact device compositions for higher-rank layouts |
| Scale and AXPY | Vectrix BLAS twins | Same exact tier |
| ReLU and erf-based GELU, forward/backward | Vectrix activation twins | Same exact tier to retain NNJava's activation semantics |
| Sigmoid, tanh and SiLU, forward/backward | Vectrix activation twins | DynamisGPU elementwise primitives |
| Counter-based dropout, forward/backward | Vectrix activation/Philox twin | Same exact tier, preserving seed/counter masks |
| Softmax, forward/backward | Vectrix softmax twin | DynamisGPU softmax primitive |
| Log-softmax, forward/backward | Vectrix stable log-softmax twin | Same exact tier |
| LayerNorm/RMSNorm and input/scale/bias gradients | Vectrix norm twins | DynamisGPU norm primitives with fixed-order parameter accumulation |
| Total and arbitrary-axis sums | Vectrix BLAS sum order | DynamisGPU fixed-tree reduction order |
| Broadcast-gradient `sumTo` | CPU serial accumulation order, composed with GPU adds | DynamisGPU fixed-tree reduction order |
| Euclidean norm | Vectrix robust BLAS norm twin | Same exact tier |
| Matrix multiplication | Vectrix GEMM twin | DynamisGPU FP32-input/FP32-output GEMM |

Matrix multiplication supports broadcast batch dimensions, transpose views,
nonzero offsets, general positive-stride matrix operands through packing, and a
zero inner dimension. Core autograd uses it for batched input/weight gradients.
Normalization and softmax operate on the last axis; the public softmax API moves
other axes through tensor views and device packing.

## Semantics and diagnostics

NNJava's GELU uses the error function. DynamisGPU's elementwise GELU uses the tanh
approximation, so the adapter selects Vectrix's erf twin in both modes. ReLU also
uses the twin to preserve the CPU signed-zero behavior. Log-softmax uses its stable
upstream operation, preserving finite log probabilities when probabilities underflow.

An all-negative-infinity softmax row has zero probabilities and zero gradient;
log-softmax produces negative infinity and zero gradient. Device reductions and
logical selection enforce this policy without reading numerical data onto the CPU.

`GpuBackend.executedTiers()` reports the selected tier per operation;
`accuracyContracts()` publishes the corresponding numerical contract.
`matmulAccuracy()` retains its convenience access to the throughput GEMM contract.
Contracts name their references: a reproducible throughput reduction can still
use a different summation order from the CPU. Exact comparisons retain upstream
subnormal/flush-to-zero limits. Per-operation contracts do not establish a
whole-model error bound.

Exact `sumTo` starts from positive zero and folds each target's contributions in
CPU input order. This can require many dispatches. Total/axis sums instead use
the BLAS tree, matching the corresponding CPU operations. No float atomics are used.

## Layouts, limits and ownership

All new numerical operations require FP32. FP16/BF16 remain storage types.
Softmax and normalization require row widths from 1 through 262,144; empty leading
dimensions are supported, including zero parameter gradients. Unsupported types
and widths fail before dispatch. Core kernel contracts reject conflicting output/input
aliasing before it reaches the numerical backend.

Inputs are packed by runtime GPU buffer copies, coalescing contiguous spans.
High-rank tensors can be flattened for kernels with lower-rank interfaces. General
packing and per-matrix batch dispatch favor correctness; they remain optimization
targets for larger workloads. Host code calculates shapes, strides and offsets.

Each recording owns its scratch handles. Closing the handles preserves native
storage retained by recorded commands; aborted encoders release those recordings.
Execution contexts retain live tensors through submission completion. Backend
closure drains work before releasing kernels and its owned device. Failure tests
inject allocation errors at six successive scratch boundaries in each mode and
check that no NNJava buffer handles leak.

## Use and validation

The [standalone GPU consumer](../examples/gpu-consumer/src/main/java/example/GpuConsumer.java)
now runs projection, bias broadcasting, SiLU, RMSNorm, softmax, loss, backward and
an AXPY weight update. It runs with the GPU adapter and native runtime, without
the concrete CPU backend, training or layer modules on its module path.

```sh
mvn -Pgpu,metal clean install
mvn -f examples/pom.xml -Pgpu clean verify
python3 tools/verify_module_consumers.py --gpu-metal
```

Hardware checks compare both modes with Vectrix CPU outputs and gradients, test
masked rows and non-last axes, multi-block norm parameter gradients, cross-batch
GEMM broadcasting, dropout counters, empty reductions and robust norms. A public
`Sequential`/`Linear`/SiLU/LayerNorm/dropout model with supplied weights executes
three explicit gradient/AXPY updates and compares every final parameter.
The [verification record](evidence/gpu-neural-2026-11-27.json) records commands and
scope. Portable builds compile this adapter and run completion tests; hardware
profiles explicitly require their selected device. Vulkan remains unverified here.

## Next GPU gates

Public mask/logic/index operations, embedding gather/scatter, RoPE,
general max/argmax and concat/slice gradients are now implemented in the subsequent
[GPU tensor-operation batch](gpu-tensor-operations.md), including masked GQA gradients.
The later [GPU model-inference batch](gpu-model-inference.md) adds supplied-weight
construction/loading and complete device-cache inference. Stateful optimizers, cross-entropy, GPU
convolution/pooling, mixed-precision computation and broader training support also
remain separate work.

Paged caches, GPU sampling, recorded decode steps and quantized placement follow
complete model operation coverage. This batch establishes GPU neural compositions
and explicit gradient updates; it does not claim a complete GPU Trainer or LLM path.

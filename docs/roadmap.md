# Neural Network Framework Implementation Roadmap

Build reusable framework capabilities on Java 25. Use Vectrix 3.0 for the CPU
implementation and the revised DynamisGPU compute runtime for GPU execution.
Application-specific model development belongs in downstream repositories.

## Current implementation

The first CPU slice now includes Maven/JDK 25, typed storage and views, basic
elementwise operations, total reductions, rank-two Vectrix GEMM, a gradient tape,
named parameters, a linear layer, MSE, and plain SGD. A deferred test backend
exercises ordered completion, retained resources, and asynchronous failures.

The build currently uses Vectrix 2.6.0 as a bootstrap dependency while 3.0 is being
prepared; the version is configurable. The exact 3.0 upgrade and the broader
operation coverage in F1/F2 remain to be completed. See the
[README](../README.md) for runnable commands and the supported API subset.

## Delivery stages

| Stage | Deliverables | Acceptance criteria |
| --- | --- | --- |
| F0 Contracts and build | Finalize package/artifact names; introduce Maven, JUnit 5, backend SPI, layout types, and an exact Vectrix 3.0.x dependency. | A clean checkout builds on JDK 25; CPU dependencies resolve by coordinates; tensor shape and backend-lifetime contracts have executable tests. |
| F1 CPU tensors | Array-backed storage, views, indexing, broadcasting, elementwise operations, reductions, and batched matrix multiplication. | Numerical reference checks pass for contiguous and supported strided layouts, empty shapes, invalid arguments, and output aliasing. |
| F2 Differentiation and modules | Gradient tape, backward rules, named/shared parameters, module composition, linear layers, embeddings, and activations. | Finite-difference checks pass; branching graphs, repeated indices, broadcast gradients, and tied parameters accumulate correctly; a custom module uses public APIs alone. |
| F3 Training and state | Losses, optimizers, parameter groups, clipping, schedules, batching interfaces, evaluation, and checkpoints. | A small example learns; optimizer updates match numerical fixtures; checkpoint/resume reproduces subsequent CPU steps with a fixed configuration and saved data position. |
| F4 Transformer primitives | Normalization, masks, attention, RoPE, residual/gated components, and KV-cache storage. | Causal and padding semantics pass; an example assembled from components has verified gradients; cached and full-sequence outputs agree within the numerical contract. |
| F5 GPU integration | Optional DynamisGPU adapter, Metal/Vulkan kernel artifacts, required primitive mappings, and device-aware tests. | Supported backends pass CPU/GPU numerical and gradient checks; a complete training step keeps intermediate state on-device; pending work retains its allocations. |
| F6 Performance and release | JMH and GPU benchmarks, allocation tuning, documentation, deployment examples, and compatibility policy. | Measured baselines exist for training and inference primitives; scalar/SIMD CPU and supported GPU configurations are documented and verified. |

F4 and F5 both build on the shared tensor and differentiation contracts. GPU
runtime development can begin at F0 using the
[DynamisGPU migration stages](dynamisgpu-rework.md#migration-and-acceptance-stages).
Full transformer GPU validation follows implementation of the relevant F4
semantics. A CPU framework release can ship before GPU support; declare the
feature set of each release explicitly.

## First implementation slice

Keep the first slice narrow enough to prove the architecture:

1. Establish Maven/JDK 25 and pin the released Vectrix 3.0.x artifact.
2. Implement FP32/INT32/BOOL metadata, CPU allocation, tensor handles, views, and
   an execution context.
3. Add fill/copy, add/multiply, sum, reshape/transpose, and matrix multiplication.
4. Implement reverse-mode differentiation for these operations, including a
   branching graph and broadcast reduction.
5. Demonstrate a linear layer and squared-error update through the public API.
6. Use the same operation descriptions in a mock asynchronous backend to verify
   completion and ownership before introducing native GPU execution.

This example verifies framework composition; its particular network is test and
example material. Keep public APIs useful for other architectures.

## Validation strategy

Use JUnit 5 for contract, numerical, gradient, and checkpoint tests. Maintain
small independently computed fixtures and double-precision finite-difference
references where useful. Pair numerical checks with integration tests that detect
missing gradient paths, accidental parameter duplication, and resource leaks.

JMH measures CPU operations and allocation. GPU benchmarks distinguish submission
overhead, transfers, execution time, and end-to-end work. Include a range of batch,
sequence, and feature sizes; record backend, device, precision, numerical mode,
and software versions. Choose performance targets after collecting baselines.

## Later capabilities

After the core contracts are stable, add convolution/pooling modules, mixed
precision with loss scaling, low-precision storage and quantization, activation
checkpointing, compiler/fusion optimizations, and additional data adapters.
Distributed training and advanced scheduling need separate designs. Downstream
LLM repositories can implement architecture-specific loaders and training recipes
using these extension points.

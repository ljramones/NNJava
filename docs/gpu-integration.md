# GPU Integration Plan

NNJava's GPU support targets the published Vectrix 4.1.0 and DynamisGPU 1.14.1 releases. Most of what this plan
originally asked of upstream now exists:

- a GPU runtime with pooled memory, command replay, strided views and profiling;
- two GPU kernel libraries;
- training kernels (backward passes, cross-entropy, optimizers) with fixed-order accumulation.

NNJava's work is the adapter: it maps tensors, autograd and modules onto those pieces, and it requests the
[remaining kernels](numerical-coverage.md#upstream-requests) upstream.

## Ownership

| Component                                                   | Responsibility                                                                                                                                                       |
|-------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| NNJava                                                      | Tensor shapes and ownership, operation semantics, differentiation, modules, builders, training, state, and the numerical adapters that choose and compose kernels.   |
| Vectrix (CPU)                                               | CPU numerical kernels: the reference semantics for each operation.                                                                                                   |
| vectrix-gpu (in the Vectrix repository)                     | **Exact** GPU twins of Vectrix kernels, bit-identical to the CPU, FP32.                                                                                              |
| DynamisGPU `dynamis-gpu-primitives`                         | **Throughput** GPU kernels (FP32/FP16/BF16/quantized), each with a published `AccuracyContract`, plus inference pieces: paged KV cache, GPU sampler, quantized GEMM. |
| DynamisGPU runtime (`dynamis-gpu-api`, `-metal`, `-vulkan`) | Devices and capability tiers, memory pools and arenas, residency budgets, queues and synchronization, command recording and replay, pipelines, profiling, errors.    |

NNJava never writes Metal, Vulkan or Slang kernels. A missing operation becomes an upstream request:

- to Vectrix, when it needs CPU semantics and an exact twin;
- to DynamisGPU, when it needs a throughput kernel.

Until a request lands, the operation is unsupported on that tier, or composed from existing kernels on the same
device. CPU fallback stays explicit.

## Two GPU modes

A GPU execution context runs in one of two modes, chosen when it is created:

- **Exact:** vectrix-gpu kernels. Results equal the CPU backend bit for bit (subnormals aside), so a training run can
  be checked against, or resumed from, a CPU run. FP32 only.
- **Throughput:** `dynamis-gpu-primitives`, falling back to exact twins where no throughput kernel exists.
  Comparisons with the CPU use each kernel's `AccuracyContract`.

Diagnostics report which tier ran each operation. Both modes keep accumulation in fixed orders; NNJava does not
enable float atomics.

## Module boundary

`nnjava-backend-vectrix-gpu` (one optional module, both modes) depends on:

- **Kernels:** `org.dynamisengine:vectrix-gpu:4.1.x` for the exact twins, and
  `org.dynamisengine:dynamis-gpu-primitives:1.14.x` for the throughput kernels.
- **Runtime:** `org.dynamisengine:dynamis-gpu-api:1.14.x` for devices, buffers, `GpuArena`, `GpuRecording`,
  `GpuSubmission`, `StridedView` and `AccuracyContract`.
- **Backends:** `dynamis-gpu-metal` and `dynamis-gpu-vulkan`, chosen by the application at runtime. The starter does
  not pull them in.

Core, layers, model definitions and CPU consumers never load these. The CPU backend depends on
`org.dynamisengine:vectrix-core` and `vectrix-ml` 4.1.x only (NNJava uses `blas`, `hash`, `core` and `nn`), not on
the `vectrix` compatibility artifact, which also brings the render and signal jars.

## Distribution

Neither library is on Maven Central. Each GitHub release carries a Maven repository ZIP with an `install.sh`:

- `vectrix-4.1.0-maven.zip` from github.com/ljramones/vectrix;
- `dynamisgpu-1.14.1-maven.zip` from github.com/ljramones/DynamicGPU.

NNJava's CI installs both before building, as DynamisGPU's CI installs Vectrix's. Two version caveats:

- **vectrix-gpu is in the Vectrix ZIP since 4.0.1.** 4.0.0 attached it only as loose jars, with no POM.
- **Versions:** vectrix-gpu 4.0.1 and 4.1.0 declare `dynamis-gpu-api` 1.13.0. The API is unchanged in 1.14.x, and
  `nnjava-bom` pins the DynamisGPU version it tests.
- **`INT64`:** the exact GELU and `GpuRope` emulate double arithmetic with 64-bit integers, so they need the device's
  `INT64` capability. Native Metal, MoltenVK and lavapipe all have it.

JVM flags:

- **SIMD:** with the `simd` classifier of `vectrix-core`/`vectrix-ml`, pass `--add-modules jdk.incubator.vector`.
- **Native access:** for Metal, `--enable-native-access=org.dynamisengine.gpu.metal` on the module path, or
  `ALL-UNNAMED` on the class path.
- **Vulkan on macOS:** it needs the SDK's loader and MoltenVK driver; see DynamisGPU's `docs/compute-runtime.md`.

Vectrix 3.x and 4.0 cannot share a build: the 3.x jar holds the same packages. The CPU migration (EXEC-01) therefore
comes before any GPU dependency.

## Compatibility gate status

1. **Artifacts and the CPU migration:** done. On 2026-11-25 the current CPU implementation compiled and passed all 54
   tests, in both the scalar and SIMD configurations, against `vectrix-core` and `vectrix-ml` 4.0.0. The only change
   was the dependency swap. EXEC-01 can land as is.
2. **Kernel coverage:** done; see [numerical coverage](numerical-coverage.md). The P0 gaps for a GPU training step
   are closed by Vectrix 4.1.0 (G-E1, G-E2, G-E3, G-E5) and DynamisGPU 1.14.0 (G-T1 to G-T6), with valid key
   lengths in `AttentionF16`'s fused kernels in 1.14.1 (G-T7).
   Open:
    - GPU convolution and pooling (G-E4, G-T8);
    - mixed-precision training (G-T9).
3. **Storage mapping:** map NNJava's `Buffer` onto `GpuBuffer`s.
    - **Allocation:** the device pools allocations internally (its SPI `MemoryPool`), and `MemoryStats` reports use
      against the residency budget.
    - **Tensor offsets and strides:** become `StridedView`s (rank 4 or less in the kit today).
    - **Per-step temporaries:** live in a `GpuArena`, reset after each step.
    - **Parameters and optimizer state:** live in pooled long-lived buffers.
    - **The 4-rank limit:** a tensor of higher rank is reshaped or made contiguous by the adapter before the call.
4. **Completion:** NNJava's `Completion` wraps DynamisGPU submissions.
    - **Waits:** `GpuSubmission.await(Duration)` and `GpuFence` give bounded waits; polling publishes readbacks.
    - **Timeouts:** a timeout that does not end the operation keeps resources retained.
    - **Closing a context:** drains its submissions first.
    - **Decode loops:** a cached decode step records once and replays through `GpuRecording`.
5. **Capabilities:** preflight a model's operations against the chosen mode, the tier table and the device's
   capability tier (`GpuCapabilities`), and report unsupported components by module path before allocation.
6. **Versions:** `nnjava-bom` pins Vectrix/vectrix-gpu 4.1.0 and DynamisGPU 1.14.1, as tested.

## Implementation and acceptance order

1. **EXEC-01:** the Vectrix 4.0 CPU migration (already verified).
2. **GPU storage:** buffers, typed copies, views, arenas and completion, on Metal, MoltenVK and lavapipe.
3. **Exact mode:** elementwise, GEMM, reductions and neural functions. Forward and backward must equal the CPU
   backend bit for bit; this is the strongest available check of the adapter itself.
4. **Throughput mode:** the same model definitions, compared within each `AccuracyContract`.
5. **Full training steps:** checkpoint and resume across CPU and exact GPU, and cached inference through
   `PagedKvCache` and the GPU `Sampler`, with intermediate state kept on the device.

Hardware runs:

- **Hardware profiles:** they fail when their device is unavailable. Portable CI runs without native initialization.
- **CI:** follow DynamisGPU's setup. It has a lavapipe container job (Linux x86_64 Vulkan in software) and a
  self-hosted Apple GPU runner (native Metal and MoltenVK).
- **Measurements:** report platform coverage as measured, and separate transfer, allocation, submission, execution
  and end-to-end costs; DynamisGPU's profiling already labels each dispatch.

The current [GPU neural-operation adapter](gpu-neural-operations.md) adds neural
functions, sums/norms, batched GEMM and gradient compositions on Metal. Full GPU
model construction/indexing/RoPE and inference orchestration remain subsequent gates.

## History

[dynamisgpu-rework.md](history/dynamisgpu-rework.md) and [dynamisgpu-commission.md](history/dynamisgpu-commission.md)
are the original requests to DynamisGPU, from its 1.0.2 era. They have been delivered and superseded. They assigned
the ML kernels to NNJava, which this plan reverses.

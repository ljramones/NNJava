# Numerical Coverage and Upstream Requests

This document maps every NNJava backend operation onto the numerical stack, and lists what is still missing
upstream. It covers:

- Vectrix 4.1.0: `vectrix-core` and `vectrix-ml` on the CPU, and `vectrix-gpu` for the GPU (4.0.1 put vectrix-gpu
  into the release ZIP);
- DynamisGPU 1.14.1: `dynamis-gpu-api` and `dynamis-gpu-primitives` (1.14.1 added valid key lengths to
  `AttentionF16`'s fused kernels).

Both were prepared on 2026-11-25 in answer to the requests below; the status of each request is in its table. Update
this document when either library releases.

The adapter now implements FP32 activations, softmax/norms/dropout, sums/norms,
batched GEMM, word masks/casts, gather/scatter, concat/slice gradients, max/argmax,
RoPE and supported gradients in both modes. See [GPU neural operations](gpu-neural-operations.md)
and [GPU tensor operations](gpu-tensor-operations.md) for layouts and semantic selections.
[GPU sampling](gpu-sampling.md) adds provider sampling and tensor-token feedback. This
coverage table describes upstream availability; it is not a claim that every row
is already wired through NNJava.

## The three numerical tiers

| Tier           | Library                                                             | Numerical contract                                                                                                                                                                             | Types                                                          |
|----------------|---------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| CPU            | Vectrix (`org.dynamisengine.vectrix.blas`, `.nn`, `.hash`, `.core`) | Fixed-order reductions. Results are recorded in a cross-platform determinism golden.                                                                                                           | FP32 (FP64 for references)                                     |
| GPU exact      | vectrix-gpu (`org.dynamisengine.vectrix.compute`, classes `Gpu*`)   | Each kernel ports the Vectrix CPU algorithm and is tested **bit for bit** against it on Metal, MoltenVK and lavapipe. The one exception is that Apple GPUs flush subnormals to zero.           | FP32                                                           |
| GPU throughput | DynamisGPU `dynamis-gpu-primitives`                                 | Each kernel publishes an `AccuracyContract` (`dynamis-gpu-api`, `compute.kit`) as data: input types, accumulation type, output rounding, order class (`FIXED` or `DEVICE`) and an error bound. | FP32, FP16, BF16 and quantized blocks, depending on the kernel |

Two rules hold in every tier:

- **No float atomics by default.** Every gradient accumulation in both GPU libraries runs in a fixed order, so results
  do not depend on scheduling. NNJava's GPU compositions must keep that property. A scatter-add must not be written
  with atomics.
- **Accuracy bounds come from the kernels.** For the throughput tier, CPU/GPU comparison tolerances come from each
  kernel's `AccuracyContract`. For the exact tier, the expected difference is zero, except for subnormals.

A GPU execution context therefore chooses its mode:

- **Exact:** CPU-identical results, FP32 only. It is slower and is the debugging and reproducibility mode.
- **Throughput:** the fast kernels, with each bound stated by the kernel.

Throughput kernels compute in a fixed order unless their contract says `DEVICE`. The choice can be made per
operation, but a context must report which tier executed each operation.

## Coverage by operation

Key:

- **yes:** the operation exists.
- **compose:** it can be built from existing kernels on the same device, with no host round trip.
- **adapter:** NNJava's adapter owns it; it is too thin for a kernel library.
- **GAP:** missing; see the requests below.

| NNJava `Operation`                                     | Vectrix CPU                                                   | GPU exact (vectrix-gpu)                                                                                 | GPU throughput (primitives)                                                                                                                                                                                                                      |
|--------------------------------------------------------|---------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| COPY, FILL, CONCAT, SLICE_BACKWARD, contiguous         | adapter                                                       | adapter: buffer copy and fill (`dynamis-gpu-api`)                                                       | yes: `Elementwise` `COPY` over `StridedView`s (rank 4 or less), plus API copy/fill                                                                                                                                                               |
| ADD, SUBTRACT, MULTIPLY, DIVIDE, SCALE (broadcast)     | `Pointwise` (4.1, contiguous), `Blasf.axpy`/`scal`            | `GpuPointwise` (4.1), `GpuBlasf.axpy`/`scal` (contiguous; the adapter materializes broadcasts)          | yes: `Elementwise` binary over broadcast views                                                                                                                                                                                                   |
| EXP, LOG, SQRT                                         | `Pointwise` (4.1)                                             | `GpuPointwise` (4.1)                                                                                    | yes: `Elementwise` unary                                                                                                                                                                                                                         |
| SUM, REDUCE_SUM, SUM_TO (broadcast-gradient reduction) | `Blasf.sum` plus adapter axes                                 | `GpuBlasf.sum` for totals/axes; serial `GpuPointwise.add` composition for SUM_TO                                                                           | `Reduce.axis` (1.14: any axis, `SUM`, `MAX`, `MIN`, `MEAN`)                                                                                                                                                                                      |
| REDUCE_MAX, MAX_BACKWARD                               | adapter                                                       | `GpuPointwise.max` (element-wise); axis max: compose with throughput tier (exact: integer-exact anyway) | `Reduce.axis` `MAX`, `Reduce.maxBackward` (1.14: first max or first NaN)                                                                                                                                                                         |
| ARGMAX                                                 | `Sampler.argmax`                                              | `GpuSampler.argmax`                                                                                     | `Reduce.argmax`                                                                                                                                                                                                                                  |
| NORM2 (global gradient norm)                           | `Blasf.nrm2`                                                  | `GpuBlasf.nrm2` per tensor, combined on the host as NNJava does                                         | `GradientClip.squaredNorm` (1.14, bit for bit)                                                                                                                                                                                                   |
| MATMUL (batched, transposes)                           | `Gemmf.gemm`                                                  | `GpuGemmf.gemm`                                                                                         | yes: `GemmF16` takes FP32, FP16 or BF16 inputs, batched, both transposes                                                                                                                                                                         |
| AXPY                                                   | `Blasf.axpy`                                                  | `GpuBlasf.axpy`                                                                                         | compose: `Elementwise`                                                                                                                                                                                                                           |
| CAST                                                   | adapter                                                       | `GpuFormats` BF16/FP8/int8 conversions                                                                  | `Cast.cast` (1.14: every pair of FP32, FP16, BF16, INT32, BOOL; integer-exact)                                                                                                                                                                   |
| EQUAL, LESS, GREATER, AND, OR, NOT, WHERE              | adapter                                                       | use the throughput kernels: they are exact on every device                                              | `Logic` (1.14): `compare`, `logical`, `not`, `select`; BOOL is a uint32 0/1 word                                                                                                                                                                 |
| IS_FINITE, ALL                                         | adapter                                                       | as above                                                                                                | `Logic.classify` (1.14: IS_FINITE, IS_NAN, IS_INF); ALL via `Reduce` over the mask                                                                                                                                                               |
| GATHER (embedding rows)                                | `Embedding.gather` (4.1)                                      | `GpuEmbedding.gather` (4.1)                                                                             | yes: `Embedding` (FP32, FP16 and quantized tables, GPU-resident tokens)                                                                                                                                                                          |
| GATHER (general axis)                                  | adapter                                                       | use the throughput kernel (a copy: exact)                                                               | `Gather.gather` (1.14)                                                                                                                                                                                                                           |
| SCATTER_ADD (embedding backward)                       | `Embedding.scatterAdd` (4.1, token order)                     | `GpuEmbedding.scatterAdd` with a host-built `Plan` (4.1)                                                | `Gather.scatterAdd` (1.14: fixed order, repeated indices in increasing position; integer counts only)                                                                                                                                            |
| RELU, SILU, GELU (tanh form) and their backward        | `Activations`                                                 | `GpuActivations`                                                                                        | yes: `Elementwise` (`RELU`, `SILU`, `GELU`, `*_BACKWARD`)                                                                                                                                                                                        |
| GELU (exact, erf form), as NNJava uses now             | `Activations.gelu`                                            | `GpuActivations.gelu`/`geluBackward` (4.1, through binary64 emulation; needs `INT64`; slow)             | still only the tanh form (erf form deferred)                                                                                                                                                                                                     |
| SIGMOID, TANH and their backward                       | `Activations`                                                 | `GpuActivations`                                                                                        | `Elementwise` `SIGMOID`, `TANH`, `SIGMOID_BACKWARD`, `TANH_BACKWARD` (1.14; the backward passes take the forward input x, the Vectrix ones the output y)                                                                                         |
| SOFTMAX, SOFTMAX_BACKWARD                              | `Softmax` (a fully masked row gives zeros since 4.1)          | `GpuSoftmax` (same rule)                                                                                | `Softmax` and `Softmax.backward` (causal mask, and since 1.14 per-row valid lengths; a fully masked row gives zeros)                                                                                                                             |
| LOG_SOFTMAX and backward                               | `Softmax`                                                     | `GpuSoftmax`                                                                                            | compose: `Reduce.logSumExp` and `Elementwise`                                                                                                                                                                                                    |
| DROPOUT                                                | `Activations.dropout` with `Philox`                           | `GpuActivations.dropout` with `GpuPhilox`                                                               | compose: the exact-tier Philox mask, then `Elementwise` `MUL`                                                                                                                                                                                    |
| LAYER_NORM, RMS_NORM, NORM_BACKWARD                    | `Norms`                                                       | `GpuNorms`                                                                                              | `Norms` and `Norms.backward` (fused residual)                                                                                                                                                                                                    |
| CROSS_ENTROPY and backward                             | `Loss`                                                        | `GpuLoss`                                                                                               | `CrossEntropy` (ignore index; the gradient may overwrite the logits)                                                                                                                                                                             |
| ROPE                                                   | `Rope`                                                        | `GpuRope` (4.1, through binary64 emulation; needs `INT64`)                                              | `Rope` (both layouts, partial rotary dimensions)                                                                                                                                                                                                 |
| ROPE_BACKWARD                                          | `Rope.applyBackward`                                          | `GpuRope.applyBackward` (4.1)                                                                           | `Rope.encodeBackward` (1.14)                                                                                                                                                                                                                     |
| Attention forward/backward (NNJava composes it today)  | `Attention` (additive mask, GQA)                              | `GpuAttention` (causal, GQA)                                                                            | `AttentionF16`: FP16 Q/K/V, causal, GQA. Since 1.14.1 the fused forward, backward and paged decode take GPU-resident valid key lengths, one per group of query heads (an empty row gives zero O, `-∞` LSE and zero gradients). No additive masks |
| SGD (momentum, Nesterov), AdamW                        | `Optim`                                                       | `GpuOptim`                                                                                              | `Optimizer` (SGD momentum, AdamW). Nesterov: check                                                                                                                                                                                               |
| Gradient clipping factor                               | `Optim.clipFactor`                                            | stays on the host (a scalar in double); read back the per-tensor norms                                  | `GradientClip.clip` (1.14: scales in place, the norm read on the GPU; within 9 ULP)                                                                                                                                                              |
| CONV2D, MAX_POOL2D, AVG_POOL2D and backward (R4, CPU implemented) | `Conv2d`, `Pool2d`: batched strided FP32 adapter, groups/depthwise, deterministic gradients | GAP (G-E4) | GAP (G-T8) |
| RNN/LSTM/GRU cells and sequences (CPU implemented; composition, no new Operation) | Existing GEMM, tanh/sigmoid, selection, views and tensor Ops; autograd unrolling | Compose upstream primitives; NNJava GPU adapter coverage pending | Compose upstream primitives; NNJava GPU adapter coverage pending |
| Random init (uniform and normal)                       | `Philox`                                                      | `GpuPhilox.fillUniform`                                                                                 | compose                                                                                                                                                                                                                                          |
| Sampling (generation)                                  | `Sampler.sample`                                              | argmax only                                                                                             | `Sampler` (greedy, temperature, top-k, top-p; the token stays on the GPU)                                                                                                                                                                        |
| KV cache                                               | NNJava `KvCache`                                              | none                                                                                                    | `PagedKvCache` (pages, fork, truncate, pin)                                                                                                                                                                                                      |
| Quantized weights (inference)                          | `BlockQuant` (llama.cpp formats), `Formats` (int8, FP8, BF16) | `GpuFormats`                                                                                            | `QuantizedMatrix`, `QuantizedGemm`, `Quantize`, `GgufBlocks`                                                                                                                                                                                     |

## Semantics to align

NNJava's [backend contract](backend-contract.md) defines several edge cases. Their state after Vectrix 4.1.0 and
DynamisGPU 1.14.1:

- **Fully masked rows:** aligned. Vectrix's `Softmax` gives zeros for an all `-∞` row (and `-∞` for log-softmax)
  since 4.1, as its GPU twins and DynamisGPU's `Softmax` do. So NNJava's CPU adapter can drop its special case.
  DynamisGPU notes that a masked row's gradient may be `-0` when the mask was additive.
- **Padded attention:** `AttentionF16`'s lengths mask keys only. Padded query rows still produce outputs and,
  if their output gradient is nonzero, gradients into K and V, so the adapter zeroes the loss gradient of padded
  positions (the ignore index does that) and keeps padded activations finite. Index the lengths as
  `[sequence][query head] / lengthGroup`: with the batch folded into the heads, `lengthGroup` = heads per sequence.
- **Max and argmax ties and NaN:** DynamisGPU 1.14's `Reduce.maxBackward` routes to the first max or the first NaN,
  as NNJava's contract says. Test `Reduce.argmaxAxis` and the samplers against the same rule.
- **Cross-entropy mean:** throughput `CrossEntropy` leaves averaging to the caller (it reports the counted rows), so
  the adapter divides by the number of valid targets.
- **GELU form:** NNJava uses the exact (erf) GELU. The exact tier has it since Vectrix 4.1; the throughput tier has
  only the tanh form, so throughput mode needs NNJava's GELU-variant setting (or the deferred erf form).
- **Sigmoid and tanh gradients:** DynamisGPU's `SIGMOID_BACKWARD` and `TANH_BACKWARD` take the forward input `x`,
  while Vectrix's `sigmoidBackward` and `tanhBackward` (and their twins) take the forward output `y`. The adapter
  passes whichever its tier expects.
- **Booleans on the GPU:** a uint32 word, 0 or 1; any nonzero word reads as true.

## Upstream requests

The ownership rule for both libraries:

- **Vectrix:** a kernel's CPU semantics live in Vectrix, and its bit-exact GPU twin lives in vectrix-gpu, changed in
  the same commit.
- **DynamisGPU:** throughput kernels live in `dynamis-gpu-primitives`, with an `AccuracyContract`.

NNJava requests each kernel from the library whose tier it needs, and never writes its own shaders.

### Vectrix (CPU semantics plus the exact GPU twin)

| ID   | Request                                                                                                                                      | Needed by                                                  | Status                                                                                               |
|------|----------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------------------------------------------------|
| G-E1 | Exact twins for: elementwise `exp`/`log`/`sqrt`/`max`, exact GELU, `Optim.clipFactor`                                                        | exact GPU mode for any training step                       | Done in 4.1: `Pointwise`/`GpuPointwise`, exact `GpuActivations.gelu`; `clipFactor` stays on the host |
| G-E2 | `nn.Embedding`: a row gather and a fixed-order scatter-add (repeated indices summed in index order) on the CPU, with twins                   | embedding training on GPU; also removes the adapter's copy | Done in 4.1: `Embedding`, `GpuEmbedding`                                                             |
| G-E3 | `GpuRope` twin of `Rope.apply`/`applyBackward`                                                                                               | exact mode for transformers                                | Done in 4.1                                                                                          |
| G-E4 | GPU twins of `Conv2d` and `Pool2d`                                                                                                           | GPU CNN examples (R4)                                      | P2, open                                                                                             |
| G-E5 | A documented fully-masked-row rule in `Softmax`/`Attention` (zeros, as NNJava's contract states), or a decision to keep the rule in adapters | consistent masking across tiers                            | Done in 4.1: zeros (`-∞` for log-softmax), CPU and GPU twins                                         |
| G-E6 | Distribution: put vectrix-gpu (its POM and jar) into the Maven repository ZIP of the Vectrix release                                         | any GPU consumer                                           | Done in 4.0.1 (also moves it to `dynamis-gpu-api` 1.13.0)                                            |

### DynamisGPU (throughput primitives)

| ID   | Request                                                                                                                                     | Needed by                                                        | Status                                                                                                                                 |
|------|---------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| G-T1 | Comparisons, logical operations and select; int32 and bool element types; casts; an is-finite mask                                          | masks, padding and any tensor program on GPU                     | Done in 1.14: `Logic`, `Cast`                                                                                                          |
| G-T2 | Reductions over any axis of a strided view (sum, max, mean), and max backward                                                               | bias gradients, broadcast-gradient reduction (`SUM_TO`), pooling | Done in 1.14: `Reduce.axis`, `argmaxAxis`, `MEAN`, `maxBackward`                                                                       |
| G-T3 | A fixed-order global squared norm over a list of buffers, and a scale-by-clip-factor update                                                 | gradient clipping without a host round trip                      | Done in 1.14: `GradientClip`                                                                                                           |
| G-T4 | A general gather along an axis, and a fixed-order scatter-add (embedding backward)                                                          | embedding training, indexing                                     | Done in 1.14: `Gather` (an out-of-range index fails the submission with `KERNEL_ERROR`)                                                |
| G-T5 | `Elementwise` `SIGMOID` and `TANH` with backward passes; optionally the erf form of GELU                                                    | LSTM/GRU in throughput mode; GELU parity                         | Done in 1.14 except the erf GELU (deferred)                                                                                            |
| G-T6 | `Rope` backward (the inverse rotation)                                                                                                      | transformer training in throughput mode                          | Done in 1.14: `Rope.encodeBackward`                                                                                                    |
| G-T7 | Padding and additive masks in `Softmax` and `AttentionF16` (per-sequence valid lengths at least), with zeros for fully masked rows          | batched training on variable-length sequences                    | Valid lengths done: `Softmax` in 1.14, `AttentionF16`'s fused forward, backward and paged decode in 1.14.1. Additive masks not offered |
| G-T8 | Convolution (implicit GEMM or im2col) and pooling, forward and backward                                                                     | GPU CNNs (R4)                                                    | P2, open                                                                                                                               |
| G-T9 | Mixed-precision training support: FP32 master weights with FP16/BF16 compute copies, loss scaling, and an overflow check feeding the scaler | EXEC-03                                                          | P2, open                                                                                                                               |

### Already present and easy to miss

- **Runtime:**
    - pooled allocation inside each device, `GpuArena` for per-step temporaries, and residency budgets reported by
      `MemoryStats`;
    - `GpuRecording` for command replay. A cached decode step is a fixed launch sequence, so replaying it avoids
      re-recording.
- **Views:** `StridedView` (offsets, strides, broadcast) for zero-copy views.
- **Accuracy:** `AccuracyContract` on every primitive.
- **Inference:**
    - `PagedKvCache`, the GPU `Sampler`, and `Embedding` by GPU-resident tokens. Together they make a decode loop whose
      logits never return to the host.
    - `GgufBlocks`, `QuantizedMatrix` and `QuantizedGemm` hold quantized weights. DynamisGPU 1.10 left the GGUF
      container reader (the file parser) to NNJava; see IO-04 in the [roadmap](roadmap.md).

# Design Review, 2026-11-25

These are the changes made to the NNJava design documents after Vectrix 4.0.0 and DynamisGPU 1.13.0 were
published. The documents were written when Vectrix had only CPU kernels and DynamisGPU was a 1.0.2 runtime. Since
then:

- **DynamisGPU 1.5–1.13** built a full compute stack:
    - a runtime with pooled memory, arenas, residency budgets, recording/replay and profiling;
    - a kernel kit with strided views and accuracy contracts;
    - throughput ML kernels, including the training backward passes.
- **Vectrix 4.0** split into four jars and gained exact GPU twins of its kernels (vectrix-gpu).

Each item below is one change: what was changed, where, and why. The code was not changed; only the documents
were. All the facts here were checked against the released code on 2026-11-25.

## 1. Two GPU kernel libraries, not one

- **Where:**
    - `architecture.md`: the repository responsibilities and the numerical policy;
    - `roadmap.md`: the responsibilities;
    - `modules.md`: the GPU module row and the ownership paragraph;
    - `backend-contract.md`: the introduction;
    - `gpu-integration.md`: rewritten;
    - `README.md`.
- **Was:** "Vectrix GPU supplies the ML kernels; DynamisGPU supplies device execution only; depend on DynamisGPU
  directly only where interop requires it."
- **Now:** the GPU has two kernel tiers:
    - **vectrix-gpu:** exact FP32 twins of the Vectrix kernels, tested bit for bit.
    - **DynamisGPU `dynamis-gpu-primitives`:** throughput kernels (FP32/FP16/BF16/quantized) with a published
      `AccuracyContract`.

  The GPU adapter depends on both, and on the DynamisGPU runtime directly.
- **Why:** this is what was built. Most of the ML kernels NNJava needs on the GPU are in `dynamis-gpu-primitives`:
    - `GemmF16` (batched, transposes, FP32 inputs accepted);
    - `AttentionF16` with backward;
    - `Norms`/`Softmax` and their backward passes;
    - `CrossEntropy` and `Optimizer`;
    - `Rope`, `Embedding`, `PagedKvCache` and `Sampler`.

  DynamisGPU's own roadmap ends with "NNJava: choosing exact (vectrix-gpu) or throughput (primitives) kernels per
  operation."

## 2. Exact and throughput GPU modes

- **Where:** `architecture.md` (numerical and resource policy), `gpu-integration.md` ("Two GPU modes").
- **Was:** "CPU/GPU comparisons use operation-specific numerical tolerances; GPU execution does not inherit a claim of
  CPU bitwise equality."
- **Now:** a GPU context runs in one of two modes:
    - **exact:** CPU-identical bits, apart from subnormals flushed on Apple GPUs;
    - **throughput:** tolerances taken from each kernel's `AccuracyContract`.
- **Why:**
    - The exact tier exists and is tested bit for bit on Metal, MoltenVK and lavapipe. It is the strongest available
      check of the GPU adapter itself: a forward and backward pass can be compared with the CPU backend for equality.
    - The throughput tier publishes its error bounds as data, so tolerances need not be guessed.

## 3. No float atomics

- **Where:** `architecture.md`, `backend-contract.md` (execution and synchronization), `numerical-coverage.md`.
- **Added:** every accumulation, including scatter-add and gradient reduction, runs in a fixed order on every
  device.
- **Why:** both upstream GPU libraries already do this. DynamisGPU 1.12 chose fixed-order accumulation over atomics
  for its backward passes, so gradients do not depend on scheduling. Float atomics are off by default across the
  stack (an owner decision). NNJava's own compositions must not reintroduce them.

## 4. FP16 and BF16 as storage types from the start

- **Where:** `architecture.md` (numerical policy), `backend-contract.md` (tensor metadata).
- **Was:** FP32, INT32 and BOOL; low precision "later".
- **Now:** FP32 compute first, but `DType` and the backend SPI carry FP16 and BF16 as storage types from the start.
  Mixed-precision training stays later (EXEC-03).
- **Why:**
    - `AttentionF16` takes FP16 Q/K/V, and `GemmF16` is fastest on FP16/BF16 inputs.
    - Adding a type to `DType` and the SPI later touches every interface and every adapter.

## 5. The GPU runtime is no longer hypothetical

- **Where:** `gpu-integration.md` (gate steps 3 and 4), `backend-contract.md` (the `Buffer` and `Completion` rows).
- **Was:** "Avoid assuming a particular upstream tensor or device-array representation before that API is
  available."
- **Now:** `Buffer` maps onto DynamisGPU's `GpuBuffer`:
    - per-step temporaries go in a `GpuArena`;
    - offsets and strides become `StridedView`s (rank 4 or less);
    - `Completion` wraps `GpuSubmission.await(Duration)` and `GpuFence`;
    - cached decode steps replay through `GpuRecording`.
- **Why:** these APIs are published in `dynamis-gpu-api` 1.13.0. Designing NNJava's SPI as a thin layer over them
  avoids a second abstraction that would then have to be fitted to them.

## 6. EXEC-01 (the Vectrix 4.0 CPU migration) is verified and comes first

- **Where:** `roadmap.md` (the baseline, the R6 row, the ordering paragraph and EXEC-01), `gpu-integration.md`,
  `modules.md`, `backend-contract.md` (dependency integration), `README.md`.
- **Verified:** a scratch copy of the current working tree was checked with only its dependency changed, from
  `vectrix` 3.0.0 to `vectrix-core` plus `vectrix-ml` 4.0.0. It compiles and passes all 54 tests in both the scalar
  and `-Psimd` configurations.
- **The change to make in `pom.xml`:**
    - replace the `vectrix` dependency with `vectrix-core` and `vectrix-ml` (the same version and classifier
      properties on both);
    - set `vectrix.version` to 4.0.0.

  Install Vectrix first from the release ZIP (`vectrix-4.0.0-maven.zip`, `sh vectrix-4.0.0/install.sh`).
- **Why first:** Vectrix 3.x and 4.0 hold the same packages, so they cannot share a build, and every GPU dependency
  is built on 4.0. Do not depend on the `vectrix` compatibility artifact; it also brings the render and signal jars,
  which NNJava does not use.

## 7. New roadmap items

- **Where:** `roadmap.md` (feature catalog).
- **Split:** EXEC-02 becomes **EXEC-02** (exact mode) and **EXEC-04** (throughput mode), because each depends on
  different upstream requests.
- **Added:**
    - **EXEC-05:** the GPU inference path: `PagedKvCache`, the GPU `Sampler`, and replayed decode steps.
    - **IO-04:** a GGUF container reader. DynamisGPU 1.10 explicitly left the GGUF file parser to NNJava; the block
      formats themselves exist (`BlockQuant` in Vectrix, and `GgufBlocks`/`QuantizedMatrix` in DynamisGPU).
- **Why:** these pieces exist upstream and were missing from the catalog.

## 8. Numerical coverage document (new)

- **Where:** `numerical-coverage.md`.
- **What:** every value of NNJava's `Operation` enum, mapped to:
    - its Vectrix CPU kernel;
    - its exact GPU twin;
    - its throughput primitive.

  Gaps carry stable request IDs: G-E1 to G-E6 for Vectrix, G-T1 to G-T9 for DynamisGPU. The document also lists
  the edge-case semantics that differ between tiers.
- **Why:** the old gate step 2 ("inventory kernel coverage") was a to-do. It is now done, and the gaps are concrete.
- **Main gaps:**
    - comparisons/select/int and bool types, and casts on the GPU;
    - axis reductions;
    - general gather and a fixed-order scatter-add (embedding backward);
    - RoPE backward;
    - sigmoid/tanh in the throughput tier;
    - padding masks in throughput attention;
    - conv/pool on the GPU.

## 9. Semantics that differ between tiers

- **Where:** `backend-contract.md` (shape and numerical conventions), `numerical-coverage.md` ("Semantics to
  align").
- **Fully masked rows:**
    - Vectrix's `Softmax` returns NaN for an all -inf row, which is why `CpuNeuralKernels` special-cases it.
    - The throughput `Softmax` gives zeros only under its built-in causal mask.
- **GELU:** NNJava uses the exact (erf) form. The throughput tier, and the exact tier's twin, have only the tanh
  form.
- **Cross-entropy mean:** the throughput `CrossEntropy` leaves the mean to the caller.
- **Why:** each GPU tier must pass NNJava's edge-case tests before it advertises an operation. These are the known
  differences to test.

## 10. Distribution and JPMS

- **Where:** `gpu-integration.md` (Distribution), `modules.md` (the dependency gate, JPMS), `architecture.md`.
- **Added:**
    - **Release ZIPs:** neither library is on Maven Central. CI installs the GitHub release ZIPs, as DynamisGPU's CI
      installs Vectrix's.
    - **vectrix-gpu:** 4.0.0 is not in the Vectrix ZIP (upstream request G-E6, planned for Vectrix 4.0.1). It declares
      `dynamis-gpu-api` 1.12.1, and the BOM pins 1.13.0.
    - **JVM flags:** native access for Metal, and `jdk.incubator.vector` for SIMD.
    - **JPMS:** module names are to be decided before R1, because the dependencies are named modules.
- **Why:** these are the practical obstacles a clean consumer or CI run hits first.

## 11. The historical DynamisGPU documents

- **Where:** `dynamisgpu-rework.md` and `dynamisgpu-commission.md` moved to `docs/history/`, each with a header note.
  Links in the README, `backend-contract.md` and `gpu-integration.md` were updated.
- **Why:** both describe DynamisGPU's 1.0.2 era. They were delivered and then superseded. They assign the ML kernels
  to NNJava, which contradicts the current ownership, and the commission is worded as an instruction to an agent.
  Kept for the record only.

## 12. Status boilerplate removed

- **Where:** every document that said "the owner has confirmed publication" or "awaiting Claude's review".
- **Now:** the status is stated once, in `roadmap.md`.
- **Why:** the review is done, and the repeated status lines would go stale everywhere at once.

## 13. Upstream requests delivered (same day)

- **Where:** `numerical-coverage.md` (coverage table, a status column on the request tables, semantics),
  `gpu-integration.md` (gate step 2, coordinates, distribution), `roadmap.md` (EXEC-02, EXEC-04).
- **Vectrix 4.0.1:** puts vectrix-gpu into the release ZIP (G-E6).
- **Vectrix 4.1.0:**
    - `Pointwise` and `Embedding` (CPU) with GPU twins `GpuPointwise` and `GpuEmbedding` (G-E1, G-E2);
    - exact GELU and `GpuRope` twins, computing in double through emulated binary64 arithmetic and needing `INT64`
      (G-E1, G-E3);
    - fully masked softmax rows give zeros (G-E5).
- **DynamisGPU 1.14.0:**
    - `Logic` and `Cast` (G-T1);
    - `Reduce.axis` and `maxBackward` (G-T2);
    - `GradientClip` (G-T3);
    - `Gather` (G-T4);
    - sigmoid and tanh with backward passes (G-T5);
    - `Rope.encodeBackward` (G-T6);
    - valid-length `Softmax` (part of G-T7).
- **DynamisGPU 1.14.1:** valid key lengths in `AttentionF16`'s fused forward, backward and paged decode (the rest
  of G-T7).
- **Still open:** the throughput erf GELU, GPU conv/pool (G-E4, G-T8)
  and mixed precision (G-T9).
- **Why it matters to the adapter:** both GPU modes now have the P0 operations of a training step, so EXEC-02 and
  EXEC-04 are adapter work only. Note the semantics differences listed in `numerical-coverage.md`: the sigmoid/tanh
  gradient argument, booleans as uint32 words, and the GELU form.

## Not changed

These were reviewed and left as written:

- the module split and its two dependency cuts;
- the lifetime, ownership and mutation rules;
- the fluent builder design;
- the examples catalog;
- keeping a particular LLM, tokenizer and recipe out of the library.

They are sound, and they do not depend on which upstream kernels exist.

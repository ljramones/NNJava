# Modular Framework and GPU Foundation

The 2026-11-27 implementation extracts the existing CPU framework into Maven and
JPMS modules, upgrades numerical dependencies, and adds an initial optional GPU
adapter. Subsequent [NN/LM builders and CPU cache sessions](model-builders.md)
are now implemented; full GPU NN/LLM execution remains roadmap work.

## Dependencies and consumption

Use JDK 25. The build pins Vectrix 4.1.0 (`vectrix-core`, `vectrix-ml`, and optional
`vectrix-gpu`) and DynamisGPU 1.14.1. Install their GitHub release Maven bundles
before building; neither numerical library is resolved from a sibling source tree.
The `simd` profile selects both Vectrix CPU classifiers and excludes the transitive
plain core jar. Applications using them enable `jdk.incubator.vector`.

The root artifact is now `org.dynamisengine:nnjava-parent:0.1.0-SNAPSHOT`, packaging
`pom`. Applications can use the `nnjava` CPU starter or select capability artifacts
through `nnjava-bom`. The old `neuralnetworks` coordinate is a compatibility starter
that transitively exposes the CPU framework. GPU dependencies remain opt-in.

| Artifact | JPMS name |
| --- | --- |
| `nnjava-core` | `org.dynamisengine.nn.core` |
| `nnjava-backend-vectrix` | `org.dynamisengine.nn.backend.vectrix` |
| `nnjava-nn` | `org.dynamisengine.nn.layers` |
| `nnjava-transformer` | `org.dynamisengine.nn.transformer` |
| `nnjava-generation` | `org.dynamisengine.nn.generation` |
| `nnjava-formats` | `org.dynamisengine.nn.formats` |
| `nnjava-tokenizers` | `org.dynamisengine.nn.tokenizers` |
| `nnjava-data` | `org.dynamisengine.nn.data` |
| `nnjava-io` | `org.dynamisengine.nn.io` |
| `nnjava-pretrained` | `org.dynamisengine.nn.pretrained` |
| `nnjava-training` | `org.dynamisengine.nn.training` |
| `nnjava` | `org.dynamisengine.nn.starter` |
| `neuralnetworks` | `org.dynamisengine.nn` |
| `nnjava-examples` | `org.dynamisengine.nn.examples` |
| `nnjava-backend-vectrix-gpu` | `org.dynamisengine.nn.backend.gpu` |

Integration tests and JMH are separate development modules. Generation is included
in the CPU starter; tokenizers remain an explicit optional dependency. The
[generation guide](generation-guide.md) documents their initial APIs and boundaries.
`nnjava-formats` parses mapped GGUF/safetensors without model or numerical modules;
IO and tokenizers now depend on formats, which uses Jackson Core 2.21.9. Minimal
IO/tokenizer module paths therefore include formats and `com.fasterxml.jackson.core`.
Generation-only consumers still need neither. See [pretrained loading](pretrained-loading.md).
`nnjava-pretrained` is an optional family-adapter artifact, outside the convenience
starter. Its standard Llama path uses IO/transformer public contracts without a
tokenizer, concrete backend or training dependency. The standalone pretrained
consumer supplies those optional inference pieces explicitly.

## Provider and persistence changes

Core no longer imports Vectrix. `ExecutionContext.cpu()` resolves the `vectrix`
provider; `ExecutionContext.open(providerId, deviceId)` selects another explicitly.
`BackendProviders.available()` lists IDs without opening devices. Providers are
registered through both JPMS and class-path service descriptors. A core-only
application gets a clear missing-provider error when requesting CPU execution.

Two experimental imports change:

- `org.dynamisengine.nn.io.TensorSnapshot` becomes
  `org.dynamisengine.nn.state.TensorSnapshot` in core.
- `org.dynamisengine.nn.io.Checkpoint` becomes
  `org.dynamisengine.nn.training.Checkpoint` for full training checkpoints.

Model-only consumers use `io.StateArchive` and `io.StateDict`, with no training/data
dependency. Archives carry neutral `state.ComponentState` records; training maps
optimizer state onto them. A binary fixture produced by the old monolithic JAR
checks version-1 checkpoint reading and byte-identical rewriting.

FP16/BF16 are storage types with raw `short[]` upload/readback and snapshots.
The CPU adapter supports storage/view copies, not low-precision arithmetic or
gradients. Original FP32/INT32/BOOL archive type IDs are preserved; archives with
the new types require the new reader. GPU BOOL storage uses 32-bit mask words.

## Initial GPU capabilities

The optional adapter accepts an owned DynamisGPU `GpuDevice` and `GpuMode`, or is
opened through providers `vectrix-gpu-exact` and `dynamis-gpu-throughput`.
It uses DynamisGPU allocation pools, private buffers, an ordered queue, and
submission completion. Readbacks publish after completion; pending bounded waits
retain resources and retry. Kernel objects are created lazily.

Implemented coverage:

- FP32, INT32, BOOL, FP16, and BF16 transfers, including empty tensors.
- FP32/INT32/BOOL view copies on-device; whole-buffer low-precision copies.
- FP32 fill, arithmetic, exp/log/sqrt, scale, AXPY, broadcast batched GEMM,
  transpose/general positive-stride views and offsets.
- Exact arithmetic through Vectrix twins; throughput arithmetic through DynamisGPU
  primitives, with scale/AXPY explicitly reported as exact-tier operations.
- Device packing for broadcast arithmetic in both modes; throughput strided
  elementwise kernels handle views through rank four directly.
- Activation, softmax/log-softmax, LayerNorm/RMSNorm and dropout forward/backward
  operations; sums, broadcast-gradient reductions and robust Euclidean norms.
- FP32/INT32/BOOL comparisons, casts, masks and selection; `all` and finite checks.
- General gather/scatter, concatenation and slice gradients, first-winner max/argmax
  and max gradients; exact RoPE forward/backward in both modes.
- Optional provider sampling: Vectrix greedy argmax and DynamisGPU temperature/top-k/top-p
  with strict device logit validation and explicit numerical contracts.

The adapter exposes selected kernel tiers and per-operation accuracy contracts.
It rejects unsupported operations/layouts before submission and never
performs implicit CPU numerical fallback. Explicit host readbacks may unpack a
view from its allocation on the host.

Stateful optimizers, cross-entropy, normal initialization and general FP16/BF16 Ops
remain unsupported through this adapter. These gates must pass before claiming
complete Trainer workflows. Supplied-weight model construction and FP32 transformer
inference now pass on Metal through the explicit loading policy; see
[GPU model inference](gpu-model-inference.md). See [GPU neural operations](gpu-neural-operations.md)
for the tested compositions, row limits and tier choices.
The [tensor-operation guide](gpu-tensor-operations.md) adds embedding/rotary/masked
GQA gradient checks, layout semantics and invalid-index completion behavior.
The [GPU sampling guide](gpu-sampling.md) covers tensor-token generation with no
logit downloads. The optional [paged inference policy](gpu-paged-inference.md)
integrates FP16 Q/K/V attention/cache storage and whole-decoder replay in throughput
mode without adding native dependencies to transformer or generation. Shared
pools, per-step arenas and controlled performance measurements remain subsequent work.

Native runtimes are supplied by the application. DynamisGPU 1.14.1's native jars
export provider constructors but do not register `GpuProvider` services; the adapter
supports both service providers and these published constructors. On the module
path, resolve the chosen native module explicitly and grant its native access.
The standalone GPU consumer demonstrates Metal selection without a CPU backend.

The published GPU shader resource packages also need qualified JPMS opens for
`KernelLibrary` in `org.dynamisengine.gpu.api`. The verification runner supplies
these flags for Vectrix's `compute.kernels` and DynamisGPU's `primitives.kernels`
and `primitives.metal` packages. Class-path execution does not require these opens.
This release integration detail should be addressed upstream in future descriptors.

## Verification evidence

The [initial 2026-11-27 verification record](evidence/modular-verification-2026-11-27.json)
reports 66 passing tests: 59 CPU/core checks, two portable completion checks, and
five Metal hardware tests exercising both modes. Scalar and SIMD reactors pass;
standalone training, model-only JPMS, and both GPU JPMS consumers pass. JMH
packaging/listing also passes. No new model-throughput benchmark is claimed.

The subsequent [GPU tensor-operation record](evidence/gpu-tensor-2026-11-27.json)
covers 288 passing tests, including the neural/indexing/rotary extensions and both
installed Metal consumers. It retains the distinction between operation coverage
and complete model inference.

The [model-inference record](evidence/gpu-model-2026-11-27.json) adds 300 passing
reactor tests and real SmolLM2 acceptance in both Metal modes, including independent
reference logits, device caches and generation. These are correctness checks;
optimized GPU serving and training remain follow-on milestones.

## Verification commands

```sh
mvn clean install                         # CPU modules, tests, examples, BOM
mvn -Psimd clean verify                   # SIMD CPU configuration
mvn -Pgpu clean verify                    # GPU compilation and portable completion tests
mvn -Pgpu,metal clean install             # required Metal hardware checks
mvn -Pgpu,vulkan clean verify             # required Vulkan hardware checks
mvn -Pbenchmarks -DskipTests package       # isolated JMH artifact
mvn -f examples/pom.xml verify            # standalone CPU consumers
mvn -f examples/pom.xml -Pgpu verify      # also compile standalone GPU consumer
python3 tools/verify_module_consumers.py --gpu-metal
```

The final command runs format-only, training, model-only, generation-only,
tokenizer-only and pretrained module-path consumers, then checks GPU provider
discovery, GEMM, neural forward/backward/updates and embedding/rotary/masked attention
in both modes. Without `--gpu-metal`, it runs the CPU consumers. Hardware profiles fail on absent devices; portable
GPU builds skip hardware tests explicitly. Local Metal evidence does not establish
Vulkan/MoltenVK coverage or real-model throughput.

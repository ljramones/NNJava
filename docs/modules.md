# NNJava Multi Module Design

Organize NNJava by capabilities that applications can select independently. Keep
model construction independent of numerical providers, and let the upstream stack
supply the mathematics: Vectrix on the CPU, and on the GPU both vectrix-gpu (exact
twins) and DynamisGPU's primitives (throughput kernels). The initial module extraction is implemented.
The [migration guide](module-migration.md) records JPMS names and current GPU limits.

## Maven modules and planned extensions

Implemented artifacts use `org.dynamisengine` and one coordinated NNJava version.
The root is the `nnjava-parent` Maven aggregator. Generation and tokenizer modules
now have initial implementations; see the [generation guide](generation-guide.md).
The optional GPU adapter implements the subset in the migration guide.

| Artifact                     | Responsibility                                                                                                                               | Internal dependencies                                                                                                                  |
|------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| `nnjava-core`                | Tensor metadata and ownership, Ops, autograd, execution contexts, backend SPI, neutral model/state contracts.                                | None                                                                                                                                   |
| `nnjava-backend-vectrix`     | CPU numerical adapter and provider registration using `vectrix-core` and `vectrix-ml` 4.1.x (not the `vectrix` compatibility artifact).      | core                                                                                                                                   |
| `nnjava-nn`                  | Parameters, modules, layer configurations, dense networks, composition, and NN builders.                                                     | core                                                                                                                                   |
| `nnjava-transformer`         | Attention, positional components, decoder/encoder building blocks, KV caches, and language-model builders.                                   | core, nn                                                                                                                               |
| `nnjava-generation`          | Token-based generation, sampling/logits processors, inference sessions, streaming, and batching/cache orchestration.                         | core, nn, transformer; no training or tokenizer dependency                                                                             |
| `nnjava-formats`             | Checked mapped GGUF/safetensors containers, shard indexes, neutral storage descriptors and shared JSON asset parsing.                         | Jackson Core; no core/model/numerical modules                                                                                           |
| `nnjava-tokenizers`          | Optional text/token interfaces, incremental decoding, byte BPE asset imports, vocabulary/special-token metadata, and providers.                 | formats; no numerical, model or training modules                                                                                        |
| `nnjava-data`                | Dataset/batch interfaces, deterministic loaders, masks, padding, packing, and data transforms.                                               | core                                                                                                                                   |
| `nnjava-io`                  | Tensor archives, model definitions/state, explicit pretrained weight mappings, model-only save/load, codecs, and schema migration.           | core, nn, formats; no numerical backend dependency                                                                                     |
| `nnjava-training`            | Losses, optimizers, schedules, metrics, Trainer, token-loss/adapter training helpers, callbacks, and full training checkpoint orchestration. | core, nn, data, io; transformer integration via public model interfaces                                                                |
| `nnjava-backend-vectrix-gpu` | Optional GPU adapter with an exact mode (vectrix-gpu) and a throughput mode (`dynamis-gpu-primitives`), on the DynamisGPU runtime.           | core; `vectrix-gpu`, `dynamis-gpu-primitives`, `dynamis-gpu-api`. The application adds `dynamis-gpu-metal` and/or `dynamis-gpu-vulkan` |
| `nnjava`                     | Convenience starter assembling the CPU framework for applications that want a single dependency.                                             | CPU capability modules above                                                                                                           |
| `nnjava-bom`                 | Align NNJava artifacts and tested numerical-provider versions. No runtime code.                                                              | Version constraints only                                                                                                               |
| `nnjava-examples`            | Runnable classical architectures, LM construction/pretraining, pretrained inference, and adaptation through public APIs.                     | Selected nn, transformer, generation, tokenizers, data, io, training, and backend modules                                              |

The GPU adapter depends on DynamisGPU directly:

- **The runtime:** buffers, arenas, submissions, recording and strided views.
- **The throughput kernels:** `dynamis-gpu-primitives`.

vectrix-gpu supplies the exact twins and itself depends only on `dynamis-gpu-api`.

The division of work:

- **NNJava:** model and tensor semantics.
- **Vectrix and vectrix-gpu:** CPU kernels and their exact GPU twins.
- **DynamisGPU:** the runtime and the throughput kernels. NNJava should request missing kernels
  upstream rather than maintain a competing ML shader collection.

A minimal inference application can select core, nn, transformer, io, and a backend.
It should not need training or dataset infrastructure. The convenience starter can
include the complete CPU feature set while leaving GPU/native dependencies opt-in.
Keep the GPU module outside the default build until its published dependencies and
required platform checks are available.

Token generation adds generation; text handling can add tokenizers or use an
application implementation. Transformer must not import generation. Format readers
return neutral metadata/tensor/block descriptors; provider-specific loaders build
GPU matrices from those descriptors, keeping io independent of the GPU adapter.
Container parsing lives in formats so metadata and tokenizer assets can be read
without loading core or model modules; see [pretrained loading](pretrained-loading.md).
The [LLM roadmap](llm-roadmap.md) defines the capability and acceptance sequence.
The first NN/LM builders live in nn/transformer, while neutral `ComponentSpec`
trees live in core and `ModelDefinitions` file IO stays backend-neutral in io.
CPU decoder cache sessions are in transformer; generation composes them without a
reverse dependency. Its initial host sampler directly uses the Vectrix core/ml
kernel artifacts with the selected scalar/SIMD classifiers, without depending on
the concrete CPU adapter. Tokenizers depend on formats/Jackson Core and remain an
explicit optional artifact; the CPU starter includes generation.

## Dependency cuts required by current code

### Core and numerical providers

`ExecutionContext.cpu()` now resolves the Vectrix service provider, removing the
core/backend cycle. Applications can also open a provider/device explicitly.
Core has no concrete numerical dependency; missing-provider and CPU isolation
checks run independently of the adapters.

Only the selected provider should initialize. Missing providers and unsupported
capabilities should produce clear errors. Importing model definitions or running a
CPU application must not initialize native GPU libraries.

Tensor, Ops, autograd, and runtime currently refer to one another. Keep them together
in core for the first extraction. Additional separation needs a demonstrated API
benefit and an acyclic design.

### State, persistence, and training

Neutral `TensorSnapshot` and `ComponentState` contracts now live in core's `state`
package. IO owns `StateArchive` and `StateDict`; training owns full `Checkpoint`
orchestration. This removes the previous training/IO cycle while retaining the
version-1 archive format.

Training checkpoint code can compose model archives and optimizer state without io
importing training. Review experimental API/package migrations explicitly; preserve
numerical behavior and saved-state meaning throughout the move.

Low-rank parameter/layer wrappers belong in nn. Training owns adapter selection,
freezing, and update orchestration; io owns neutral adapter archives. Loading or
merging a trained adapter for inference must not require training infrastructure.

### Tests and examples

Core tests must not depend on a numerical adapter that itself depends on core.
Use a minimal fake backend within core tests. Place numerical integration tests
in adapter or integration-test modules. A shared test-support artifact can depend
on core and be consumed by adapters; core must not depend back on it.

## Development modules

- `nnjava-integration-tests`: consumer-level numerical, gradient, checkpoint,
  ownership, and complete model-construction tests. Not a runtime dependency.
- `nnjava-benchmarks`: JMH and later device measurements, isolated from production
  class output and normal dependency resolution.

Include `nnjava-examples` in the reactor during R1, beginning with existing small
runnable examples. Adopt the fluent API in R2/R3 and add CNN/recurrent families as
their primitives arrive in R4. Production libraries never depend on examples.
The [architecture catalog](examples.md) defines its tasks and acceptance checks.

Introduce other development modules as their responsibilities are extracted.
Avoid empty placeholders. Keep the existing standalone consumer as a separate
check that published public APIs are sufficient outside the reactor.

## Build and compatibility policy

Use JDK 25 and standard Maven source layouts. Decide JPMS names and package
ownership before R1, then implement descriptors alongside the module extraction
after checking provider discovery.
Avoid split packages and dependency cycles, including test-scoped cycles.

Consider a temporary compatibility starter for the current `neuralnetworks`
coordinate. New artifact naming, package changes, and checkpoint-format migration
need release notes before a stable release. BOM/starter dependencies must not add
native libraries to CPU-only consumers.

## Numerical dependency gate

The CPU build uses Vectrix 4.1.0; the optional GPU adapter pins DynamisGPU 1.14.1.

- **The CPU migration:** the earlier `vectrix-core`/`vectrix-ml` 4.0.0 scratch
  check passed. EXEC-01 and the module extraction are now implemented and
  verified against 4.1.0, including the SIMD configuration.
- **Coordinates, coverage and gaps:** in [GPU integration](gpu-integration.md) and
  [numerical coverage](numerical-coverage.md).
- **Distribution:** neither library is on Maven Central. CI installs their release
  ZIPs, and the BOM pins the tested versions.

Model and builder design can proceed independently of GPU integration. The
[main roadmap](roadmap.md) records implemented milestones and remaining work.

JPMS names were decided and implemented during R1; see the migration guide. The numerical dependencies are named
modules: `org.dynamisengine.vectrix`, `org.dynamisengine.vectrix.ml`,
`org.dynamisengine.vectrix.compute` (vectrix-gpu) and `org.dynamisengine.gpu.*`.

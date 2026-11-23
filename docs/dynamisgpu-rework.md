# DynamisGPU Compute and Metal Rework

DynamisGPU should provide a general GPU execution foundation used by graphics,
simulation, and this neural-network framework. Add native Metal alongside Vulkan
and introduce compute services that can run without windows, swapchains, frame
loops, or mesh resources.

Status: proposed changes for the DynamisGPU repository. This plan defines the
boundary needed by NeuralNetworks; implementation of the rework belongs in
DynamisGPU.

Implementation instructions and acceptance criteria are in the
[execution commission](dynamisgpu-commission.md).

## Current integration constraints

The current module layout is `dynamis-gpu-api`, `dynamis-gpu-vulkan`,
`dynamis-gpu-test`, and `dynamis-gpu-bench`. Its README centers on geometry upload,
staging, and mesh resources. macOS Vulkan setup uses MoltenVK.

Several public contracts carry graphics or Vulkan assumptions:

- `gpu.VkCommandBuffer` exposes a Vulkan-named native handle at API level.
- `gpu.StagingScheduler.flush` accepts that type and its reset contract refers to
  frames.
- `upload.UploadTicket.await` returns a `GpuMeshResource`.
- `buffer.GpuBufferUsage` expresses one usage value, although a compute allocation
  commonly needs storage plus transfer-source and transfer-destination usage.
- `VulkanFrameSyncResources` builds synchronization around frames and presentation.

`VulkanComputeCullingDispatch` currently captures submitted values; it does not
implement a general shader pipeline and dispatch service. Its existence should
not be treated as completion of the compute backend.

The parent POM currently declares Java 21 source/target and the Vulkan module
includes macOS ARM64 LWJGL natives. Plan and test the revised runtime on JDK 25;
choose the supported OS/architecture matrix explicitly when updating its build.

## Responsibility boundary

| DynamisGPU owns | NeuralNetworks owns |
| --- | --- |
| Devices, memory, transfers, queues, synchronization, executable pipelines, bindings, dispatch, and native errors. | Tensor shapes, strides, broadcasting, differentiation, modules, optimizers, and training state. |
| Loading Metal libraries or Vulkan shader binaries and reporting hardware capabilities. | ML kernel sources/artifacts, specialization choices, operation composition, and numerical validation against Vectrix. |
| Generic allocation pools, resource retirement, and GPU timing. | Tensor memory planning, saved activations, KV-cache organization, and model-level profiling. |

This keeps the GPU runtime usable by other domains. Backend-specific ML kernels
can live inside the framework's GPU adapter. A future reusable GPU numerical
kernel library can be extracted when there is a second consumer.

## Module changes

Extend `dynamis-gpu-api` with neutral `device`, `memory`, `queue`, `compute`, and
`sync` contracts. Keep those contracts free of LWJGL, Metal, mesh, and frame types.
Existing graphics contracts can delegate through compatibility adapters while
their implementations adopt the new foundation.

Add `dynamis-gpu-metal` for the native Metal implementation and its bridge. Extend
`dynamis-gpu-vulkan` with headless compute using the same contracts. Expand
`dynamis-gpu-test` into a backend conformance suite and add compute workloads to
`dynamis-gpu-bench`. Separate graphics extensions into their own artifact later
when that separation has a clear migration path.

## Required generic APIs

| Proposed type | Required behavior |
| --- | --- |
| `GpuProvider` and `GpuAdapter` | Enumerate usable hardware, report backend identity and limits, and open a headless device. |
| `GpuDevice` | Own allocations, queues, pipelines, and capability queries; validate device ownership of resources. |
| `GpuQueue` | Submit ordered work and completion dependencies without requiring a render frame. |
| `GpuBuffer` and `BufferSlice` | Support byte sizes, alignment, combinable usages, placement intent, explicit mapping, and retained subranges. |
| `GpuCommandEncoder` | Record buffer copies, fills, compute bindings, dispatches, and resource access declarations. |
| `GpuComputePipeline` and `BindingLayout` | Load a compute entry point with explicit buffer bindings, parameter layout, specialization values, and workgroup requirements. |
| `GpuSubmission` | Report completion/failure, expose a bounded wait or polling API, and retain resources until completion. |
| `GpuCapabilities` | Report limits, shader features, supported scalar/storage types, timestamp support, and optional optimized capabilities. |

Use opaque typed resources in common APIs. Put native-handle interop in an
explicit backend extension. Identify backends separately: native Metal and
Vulkan through MoltenVK are distinct execution paths.

Start with one ordered compute queue. Define read/write/transfer access and
hazards between commands; submission order alone is not a substitute for memory
visibility. Backend implementations translate those rules into Vulkan barriers
and Metal encoder/resource synchronization. Add multiple queues only after their
dependency semantics are covered by tests.

## Buffer and transfer changes

Replace single-value usage descriptions with a usage set. Support storage,
transfer source/destination, and uniform/parameter data where applicable. Separate
usage from placement and access policy.

Offer upload, download, device copy, fill, and mapped host access with checked
byte ranges and explicit completion. On Vulkan, account for coherent versus
noncoherent host memory and queue-family ownership when relevant. On Metal,
support shared and private storage first; expose managed storage only if the
supported Mac matrix needs it and its synchronization is implemented.

Apple Silicon unified memory still requires ownership and completion rules.
Moving Java heap arrays into a GPU-visible buffer is not automatically zero-copy.
Document when mapping or copying occurs and keep native mappings within their
declared lifetime.

Generalize the upload arena and deferred retirement ideas into reusable pools.
Retain every allocation used by an in-flight submission and return it to the pool
only after completion. Allocation failure must surface as a resource error the
framework can report, with no partially usable tensor returned.

## Native Metal implementation

Use JDK 25's Foreign Function and Memory API with a small Objective-C or
Objective-C++ bridge exposing a stable C ABI. Keep native binding details inside
the Metal module. The bridge wraps:

- Device discovery and capabilities using `MTLDevice`.
- Queue and command-buffer creation using `MTLCommandQueue` and `MTLCommandBuffer`.
- Buffer allocation, mapping, and blit transfers using `MTLBuffer`.
- Library/function loading and `MTLComputePipelineState` creation.
- Compute encoding, binding, dispatch, completion, and error reporting.

Use opaque handles with explicit retain/release ownership. Keep Objective-C
exceptions inside the bridge and return structured error codes and copied
messages. Manage autorelease pools at native entry boundaries. Java FFM arenas
own call-side memory; Metal resources also need their own native retention until
GPU completion.

Begin with polling/status queries and explicit waits for completion. Native
completion callbacks can follow once their thread and Java callback lifetimes
are defined. Package native artifacts by supported architecture, with a versioned
bridge ABI and a reproducible Apple-toolchain build. Document the native-access
JVM flag for the proposed named module,
`--enable-native-access=org.dynamisengine.gpu.metal`, or `ALL-UNNAMED` when used on
the class path. Build the native bridge and Metal libraries with the Xcode tools
on macOS and publish matching native artifacts. Portable Java builds on other
platforms should resolve their selected provider without compiling Metal sources.

Validate native Metal independently of MoltenVK. Keep provider selection
configurable and report the actual backend/device in diagnostics and benchmarks.

## Shader and pipeline portability

A portable compute API can consume backend-specific executables. Package a
logical kernel with a binding schema, parameter-byte layout, and implementations
for SPIR-V/Vulkan and Metal libraries. For Metal, build MSL into a `metallib` for
the supported deployment target; optional runtime compilation can serve developer
iteration.

Define buffer binding slots, scalar widths, alignment, and parameter structure
padding explicitly. Use dispatch group counts and an explicit local group size;
pass logical tensor sizes separately and guard partial groups in the kernel.
Validate local-size limits and required features before creating the pipeline.

Cache pipelines using the kernel version, executable identity, device/backend,
specialization values, and numerical options. Preserve compile diagnostics.
Treat MSL and SPIR-V as separate artifacts initially; shader translation is a
future tooling choice with its own validation burden.

Expose optional optimized paths such as Metal Performance Shaders through a
capability extension once measured. Baseline compute must work without that
extension. FP32 comes first; FP16, BF16, subgroup operations, integer atomics, and
specialized matrix instructions require individual capability checks. Do not make
FP64 a requirement for a neural-network-capable device.

## Changes to Vulkan execution

Factor device creation and compute submission out of graphics/frame setup. Select
a compute-capable queue; a graphics-capable queue is acceptable when it supports
compute, but presentation support must not be required.

Implement real compute pipeline creation, descriptor/binding allocation, dispatch,
buffer barriers, and asynchronous completion. Reuse lower-level buffer and fence
utilities where their contracts fit. Helpers that currently wait for an entire
queue after a transfer should gain submission-scoped completion for sustained
compute workloads. Keep frame synchronization as a graphics consumer.

## Required neural framework workloads

The framework adapter should first prove FP32 copies, elementwise operations,
reductions, matrix multiplication, gather/scatter-add, and their gradients. Follow
with normalization, softmax, activations, loss, optimizer updates, RoPE, and
attention. The full operation list is in the
[backend contract](backend-contract.md#required-operation-groups).

Keep parameters, gradients, optimizer state, and intermediate values on the GPU
across a training step. Host transfers happen at declared data and observation
boundaries. Support both matrix-heavy sequence processing and the smaller
operations used during cached decoding. Batching commands reduces Java/native
transition costs and avoids synchronizing after each primitive.

## Migration and acceptance stages

| Stage | Change | Acceptance evidence |
| --- | --- | --- |
| G0 | Introduce neutral devices, buffers, queues, bindings, completion, and capability contracts. | Mock contract tests cover ordering, validation, ownership, and deferred release; existing graphics tests continue to pass. |
| G1 | Implement native Metal discovery, allocation, transfers, pipeline loading, and dispatch. | A headless vector-add workload executes on Metal and returns verified results; tests exercise completion and errors. |
| G2 | Adapt Vulkan to the same compute contract. | The same copy/fill/vector-add conformance suite passes on supported Vulkan hardware, without presentation setup. |
| G3 | Implement framework GPU primitives through the runtime. | GEMM, backward operations, and one optimizer step agree numerically with CPU references; data stays on-device through the step. |
| G4 | Tune allocation, dispatch batching, GEMM, and selected fused kernels. | Benchmarks separate native-call overhead, transfers, execution, peak allocation, and end-to-end throughput. |

Run actual Metal and Vulkan hardware tests in addition to mocks. Include repeated
allocation/release, close-before-completion, invalid bindings, non-divisible
dispatch sizes, readback ordering, and shader errors. GPU-required CI jobs should
fail if their required device is unavailable; portable jobs can report an explicit
hardware-test skip.

## First changes to make in DynamisGPU

1. Add neutral compute contracts and a conformance suite inside the existing API
   and test modules.
2. Introduce the Metal module, bridge build, and headless copy/vector-add example.
3. Move reusable allocation and submission logic beneath the graphics adapters.
4. Add real Vulkan compute execution against the same tests.
5. Publish a versioned runtime contract for the NeuralNetworks GPU adapter.

These changes establish the GPU foundation while the framework develops its
tensor and CPU implementation. Their integration point is the shared compute
contract, rather than a dependency on geometry-upload services.

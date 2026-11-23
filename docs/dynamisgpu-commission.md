# DynamisGPU Compute and Metal Implementation Commission

Implement a general compute foundation in DynamisGPU, including a native Metal
backend and headless Vulkan compute. Preserve the existing graphics use cases
through compatible APIs or migration adapters. Deliver working code, tests,
build support, and documentation against JDK 25.

This is an execution assignment. Inspect the current repository, establish a
baseline, and proceed through the implementation stages below. Resolve routine
implementation choices and record material design deviations with their reasons.

## Working repository and required reading

Work in:

```text
/Users/larrymitchell/Dynamis/Layer2-Geometry-Animation/DynamisGPU
```

Read that repository's `AGENTS.md` and applicable directory instructions, then:

1. [DynamisGPU compute and Metal plan](dynamisgpu-rework.md).
2. [NeuralNetworks tensor and backend contract](backend-contract.md).
3. [NeuralNetworks architecture](architecture.md), particularly repository responsibilities.

These documents are located in:

```text
/Users/larrymitchell/Dynamis/AI-extra/NeuralNetworks/docs
```

Inspect the current source before applying the plan's observations; other work
may have advanced since they were written. Preserve unrelated changes and build
on working implementations already present. Keep implementation changes within
DynamisGPU. The NeuralNetworks documents provide the consumer contract.

## Required ownership boundaries

DynamisGPU owns devices, buffers, transfers, queues, resource synchronization,
compute pipelines, bindings, dispatch, capabilities, native errors, and generic
profiling. Its public compute API must serve graphics, simulation, and numerical
applications without requiring mesh resources or render frames.

NeuralNetworks owns tensors, automatic differentiation, network layers, training,
and ML-specific GPU kernels. The Vectrix 3.0 CPU integration also belongs to
NeuralNetworks. DynamisGPU's runtime must be buildable and usable independently
of either library. Generic test shaders such as vector addition belong in the
GPU conformance suite.

Execute the runtime work in plan stages G0, G1, and G2. Supply the runtime hooks,
diagnostics, and documentation needed for later G3/G4 integration. ML kernels and
the NeuralNetworks adapter are separate deliverables in their owning repository.

## Stage 1 Establish the build and migration baseline

- Inspect module dependencies, public APIs, native bindings, existing tests, and
  current platform assumptions. Record the baseline test results.
- Set the revised build and tests to JDK 25. Check inherited compiler settings
  and use one consistent release target. Update setup documentation accordingly.
- Identify existing graphics consumers and retain their usable contracts through
  compatibility adapters as the compute foundation is introduced.
- Use the current Maven module structure. Add `dynamis-gpu-metal`; introduce
  further module splits only when needed to maintain a clear dependency boundary.

Acceptance: the existing Java/Vulkan modules build on JDK 25, baseline regressions
are identified, and the migration preserves existing graphics behavior.

## Stage 2 Implement the generic compute contract

Extend `dynamis-gpu-api` with neutral contracts for providers/adapters, devices,
buffers and slices, queues, command encoders, compute pipelines, binding layouts,
submissions, and capabilities. Proposed names in the plan may be adapted to
established repository conventions.

The contract must provide:

- Headless device discovery and explicit provider/device selection.
- Checked byte ranges and alignment, combinable buffer usages, allocation
  placement, upload/download, fill, copy, and supported mapping behavior.
- Pipeline entry points, buffer binding ranges, scalar parameter layouts,
  specialization options, and validated dispatch/workgroup sizes.
- One ordered compute queue initially, with explicit memory access and visibility
  rules for transfer and compute dependencies.
- Pollable completion, bounded waiting, structured errors, and explicit resource
  ownership. In-flight work retains allocations, bindings, pipelines, and upload
  data until their use completes.
- Capability reporting for storage limits, data types, shader features, workgroup
  limits, and optional timing. Unsupported requests fail with actionable details.

Keep Vulkan/Metal native handles and binding classes in backend extensions. The
generic contracts must work without a swapchain, presentation queue, frame index,
or mesh payload. Existing `VkCommandBuffer` and geometry upload contracts can
remain as graphics compatibility surfaces.

Implement a shared JUnit 5 conformance suite and a mock backend. Acceptance:
contract tests prove ordering, validation, failure propagation, alias/slice
ownership, idempotent close, and release after pending work completes.

## Stage 3 Implement native Metal

Implement `dynamis-gpu-metal` through JDK 25's Foreign Function and Memory API and
a small Objective-C or Objective-C++ bridge with a versioned C ABI.

- Wrap Metal device discovery, queues, buffers, blit transfers, libraries,
  functions, compute pipelines, command encoding, and submission status.
- Use typed opaque handles and explicit native ownership. Scope autorelease
  pools, contain Objective-C exceptions, and copy error messages across the ABI.
- Implement shared and private storage. Document mapping lifetimes and CPU/GPU
  visibility; shared memory still requires completion synchronization.
- Start completion handling with polling and explicit waits. Keep FFM arena and
  native resource lifetimes valid through asynchronous submissions.
- Build the bridge and MSL test shaders on macOS with the Apple toolchain.
  Package the native library and Metal executables for the supported architecture
  and deployment target. Validate bridge ABI compatibility when loading.
- Document native-access JVM flags and provide an explicit Metal test profile.
  Keep ordinary builds on other platforms independent of Metal compilation.

Acceptance: a headless upload, native Metal vector-add dispatch, and download
round trip returns verified results. Tests exercise shared/private buffers,
binding validation, partial workgroups, completion, deferred release, and native
errors. Diagnostics identify native Metal and the actual device. Execution through
MoltenVK does not satisfy this stage's native Metal acceptance criteria.

## Stage 4 Implement headless Vulkan compute

Adapt the Vulkan backend to the same public compute contracts and conformance
tests. Factor reusable memory and submission services out of graphics setup.

- Open a compute-capable device and queue without presentation requirements.
- Implement actual compute pipeline creation, shader binding, buffer dispatch,
  transfer/compute barriers, submission, and error reporting.
- Account for host-memory coherency and queue ownership where applicable.
- Use submission-scoped completion in sustained compute paths. Keep frame
  synchronization available to graphics consumers.
- Reuse lower-level allocation and synchronization utilities where their
  behavior matches the generic contract.

Acceptance: the shared copy/fill/vector-add suite passes on a supported Vulkan
device. Existing geometry-upload and graphics tests continue to pass. A class
that only records dispatch arguments does not satisfy GPU execution acceptance.

## Stage 5 Complete integration documentation and evidence

Provide a small public-API example that selects a provider, allocates buffers,
loads a compute pipeline, binds inputs/outputs, dispatches work, awaits completion,
reads the result, and closes resources. Keep its control flow common to Metal and
Vulkan; select the appropriate shader executable through a kernel bundle.

Document the shader binding/parameter ABI, pipeline caching identity, resource
lifetimes, numerical feature reporting, and platform support matrix. Demonstrate
consecutive dependent GPU operations without an intermediate host round trip.
Add a generic compute benchmark that distinguishes submission, transfer, execution,
and allocation costs. Record backend, device, software versions, and workload size.

Update the README and add a durable implementation record inside DynamisGPU's
`docs/` directory. Include delivered APIs/modules, exact build/test commands,
hardware results, compatibility notes, and remaining integration work. Provide a
handoff summary that the NeuralNetworks session can use to implement its adapter.

## Verification requirements

Start with the repository's existing commands:

```bash
java -version
mvn clean verify
mvn -pl dynamis-gpu-vulkan -am test
```

Introduce and document the exact commands for native Metal builds, Metal hardware
tests, Vulkan hardware tests, and compute examples as those paths are implemented.
Ensure forked test JVMs receive the native-access flags they need.

Use mocks for portable contract checks and real hardware for backend acceptance.
Test out-of-range bindings, cross-device misuse, partial dispatch groups, shader
failures, pending-operation readback, repeated allocation/release, and close before
completion. Where timeout or device-loss behavior can be injected, verify failure
propagation and resource cleanup without requiring a real device failure.

Hardware-required validation must fail if its required backend is unavailable;
portable test runs may explicitly skip hardware tests. If hardware or toolchain
access blocks a check, continue independent implementation and report the exact
unverified behavior. Distinguish mock success, successful compilation, real GPU
execution, and benchmark measurements in the handoff.

## Completion criteria

The commission is complete when JDK 25 builds, the generic compute API, native
Metal execution, headless Vulkan execution, compatibility coverage, conformance
tests, examples, and integration documentation are implemented and verified on
their declared platforms. Clearly identify any acceptance criteria still awaiting
hardware or external prerequisites; those criteria remain incomplete.

Begin with repository inspection and baseline checks, then implement the stages
in order. Continue through validated stages without adding approval gates for
routine implementation choices. Deliver the working changes and verification
record for review.

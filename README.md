# NNJava

A Java 25 library for constructing, training, and executing neural networks and
transformers. The CPU implementation uses your Vectrix library for numerical
kernels. Applications choose their architectures, weights, vocabulary, data, and
training recipes in separate projects. Planned LLM support includes reusable
generation, optional tokenizers, weight loading, pretraining, and fine-tuning.

## Build and run

Requirements: JDK 25+, Maven, and Vectrix available through Maven.

```sh
mvn clean verify                              # CPU correctness and integration tests
mvn -Psimd test                               # Vectrix SIMD configuration
mvn install                                   # Install modules, CPU starter, and BOM
mvn -f nnjava-examples/pom.xml exec:java -Dexec.args=--list
mvn -f nnjava-examples/pom.xml exec:java -Dexec.args=linear-regression
mvn -f examples/pom.xml verify                 # Independent training and inference consumers
python3 tools/verify_module_consumers.py       # Model-only consumer on the module path
```

CPU starter: `org.dynamisengine:nnjava:0.1.0-SNAPSHOT`. The previous
`neuralnetworks` coordinate remains a compatibility starter. Use `nnjava-bom` to
select capability modules. Packages live under `org.dynamisengine.nn`; sources use
standard Maven layouts inside each module. See the [module migration guide](docs/module-migration.md)
for JPMS names, experimental import changes, and model-only consumption.

The build pins **Vectrix 4.1.0**, using `vectrix-core` and `vectrix-ml` from the
published Maven release bundle. The optional GPU adapter uses `vectrix-gpu` and
**DynamisGPU 1.14.1**. There is no absolute source-tree dependency. The `simd`
profile selects both CPU classifiers and excludes the transitive plain core jar;
test JVMs enable `jdk.incubator.vector`.

The framework is a Maven reactor with explicit JPMS production modules, an
examples module, integration tests, and optional benchmarks/GPU adapters. The
[roadmap](docs/roadmap.md), [module design](docs/modules.md),
[fluent builder proposal](docs/fluent-builders.md), and
[examples catalog](docs/examples.md) describe implemented and planned capabilities.

## Implemented CPU support

| Area | Components |
| --- | --- |
| Tensors | FP32/INT32/BOOL plus FP16/BF16 raw storage, checked shapes, views, positive-step slices, permutation, broadcasting, gather/scatter, concatenation, and casts. |
| Mathematics | Elementwise functions, masks/selection, total/axis reductions, vector norms, and broadcast batched Vectrix GEMM. |
| Differentiation | First-order reverse mode, shared-input accumulation, broadcast/index gradients, saved-storage mutation checks, and explicit lifetime handling. |
| Networks | Recursive modules and parameters, Sequential, Linear, activations, embeddings, dropout, LayerNorm/RMSNorm, residuals, and tied weights. |
| Training | MSE/cross-entropy, momentum/Nesterov SGD, AdamW, groups, schedules, finite checks, clipping, accuracy, datasets, and deterministic batching. |
| Persistence | Named state, aliases, module configuration/modes, optimizer slots/counters, random counters, loader metadata, CRC validation, and atomic replacement. |
| Transformers | Scaled/grouped-query attention, causal/padding masks, rotary positions, SwiGLU, multi-head projections, and inference KV caches. |

Tests include independent numerical references, gradient checks, nonlinear XOR
training, caller-defined transformer training, exact checkpoint resume, cached
attention parity, and a deferred backend for ordering/failure cleanup.

## Compose a network

```java
try (var context = ExecutionContext.cpu();
        var model = new Sequential(context,
                new Linear(context, 2, 8, 11), Activation.tanh(context),
                new Linear(context, 8, 2, 12));
        var optimizer = new AdamW(model.namedParameters(), 0.03f);
        var x = context.tensor(new float[] {0,0, 0,1, 1,0, 1,1}, Shape.of(4, 2));
        var labels = context.tensor(new int[] {0, 1, 1, 0}, Shape.of(4))) {
    for (int step = 0; step < 250; step++) {
        optimizer.zeroGrad();
        try (var tape = context.gradientTape();
                var logits = model.forward(x);
                var loss = Losses.crossEntropy(logits, labels)) {
            tape.backward(loss);
        }
        optimizer.step();
    }
}
```

See the [complete standalone consumer](examples/consumer/src/main/java/example/LibraryConsumer.java)
for public imports and a transformer assembled outside the library's source tree.
`AbstractModule` supplies custom module registration and ownership. Each returned
tensor is an owned handle; close it or retain another handle explicitly. Contexts
are thread-confined, and inference mode is independent of gradient recording.

## Benchmarks and CI

```sh
mvn -Pbenchmarks -DskipTests package
java -jar nnjava-benchmarks/target/benchmarks.jar CoreBenchmark -prof gc
```

JMH measures public-API execution/readback and training-step cost, including
allocation. Initial measurements and their limits belong in
[CPU verification evidence](docs/cpu-verification.md). CPU CI installs the published Vectrix bundle, checks scalar/SIMD execution, and
runs standalone consumers. Portable GPU CI compiles the optional adapter against
the published DynamisGPU bundle and checks completion contracts. Hosted CI results are separate from local verification.

## GPU integration status

The optional GPU foundation has two modes:
- **Exact:** vectrix-gpu's twins of the Vectrix kernels, bit-identical to the CPU.
- **Throughput:** DynamisGPU's primitives, with published accuracy contracts.

Both run on DynamisGPU's runtime (Metal and Vulkan). NNJava owns tensor/model
semantics and the adapters. Typed storage, an initial FP32 arithmetic subset, and
rank-two GEMM have been exercised locally on Metal in both modes. Full NN/LLM
inference and training require further operation coverage; Vulkan execution is not
yet verified here. See the [implemented capability limits](docs/module-migration.md#initial-gpu-capabilities).

```sh
mvn -Pgpu,metal clean install
mvn -f examples/pom.xml -Pgpu verify
python3 tools/verify_module_consumers.py --gpu-metal
```

The [integration plan](docs/gpu-integration.md) records
ownership, distribution and the remaining gates, and
[numerical coverage](docs/numerical-coverage.md) maps every operation to its
kernels and lists the upstream requests. CPU users have no native GPU dependencies.

## Plans and guides

- [CPU API guide](docs/cpu-guide.md)
- [Implemented module migration and GPU foundation](docs/module-migration.md)
- [Implementation roadmap and status](docs/roadmap.md)
- [LLM capabilities and delivery milestones](docs/llm-roadmap.md)
- [Multi-module design](docs/modules.md)
- [Fluent builder proposal](docs/fluent-builders.md)
- [Classical architecture examples](docs/examples.md)
- [Framework architecture](docs/architecture.md)
- [Tensor and backend contract](docs/backend-contract.md)
- [GPU integration plan](docs/gpu-integration.md)
- [Numerical coverage and upstream requests](docs/numerical-coverage.md)
- [Design review of 2026-11-25](docs/design-review-2026-11-25.md)
- History: the original [DynamisGPU rework plan](docs/history/dynamisgpu-rework.md)
  and [execution commission](docs/history/dynamisgpu-commission.md)

Convolution/pooling, RNN/LSTM/GRU layers, fluent creation APIs, and their runnable
architecture examples are planned in the roadmap. Mixed precision,
nested/higher-order differentiation, graph retention, distributed execution, and
further performance work remain later extensions. A production LLM and its
application behavior belong in the downstream model repository.

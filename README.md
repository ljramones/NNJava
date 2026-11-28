# NNJava

A Java 25 library for constructing, training, and executing neural networks and
transformers. The CPU implementation uses your Vectrix library for numerical
kernels. Applications choose their architectures, weights, vocabulary, data, and
training recipes in separate projects. Fluent NN/causal-model definitions and CPU
cache sessions, fluent training, token generation, and optional byte-level BPE
are implemented, alongside general GGUF/safetensors readers, explicit weight mappings,
and GPT-2/selected ByteLevel tokenizer-file loading. An optional Llama adapter now
runs SmolLM2-135M against independent tokenizer/logit references on scalar and SIMD CPU.
GPU model execution, pretraining data workflows, and fine-tuning remain on the roadmap. A separate experimental GGUF/Metal path is
described below.

## Build and run

Requirements: JDK 25+, Maven, and Vectrix available through Maven.

```sh
mvn clean verify                              # CPU correctness and integration tests
mvn -Psimd test                               # Vectrix SIMD configuration
mvn install                                   # Install modules, CPU starter, and BOM
mvn -f nnjava-examples/pom.xml exec:java -Dexec.args=--list
mvn -f nnjava-examples/pom.xml exec:java -Dexec.args=linear-regression
mvn -f nnjava-examples/pom.xml exec:java -Dexec.args=mlp-xor
mvn -f nnjava-examples/pom.xml exec:java -Dexec.args=causal-lm
mvn -f nnjava-examples/pom.xml exec:java -Dexec.args=autoencoder
mvn -f nnjava-examples/pom.xml exec:java -Dexec.args=cnn
mvn -f nnjava-examples/pom.xml exec:java -Dexec.args=rnn
mvn -f nnjava-examples/pom.xml exec:java -Dexec.args=lstm
mvn -f nnjava-examples/pom.xml exec:java -Dexec.args=gru
mvn -f nnjava-examples/pom.xml exec:java -Dexec.args=text-generation
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
| Fluent construction | Immutable sequential/Linear and causal-decoder definitions, versioned definition files, named components, inferred sample shapes, reproducible seeds, and pre-allocation validation. |
| Networks | Recursive modules, Linear, grouped/depthwise Conv2d, pooling, RNN/LSTM/GRU cells and sequences, explicit recurrent state/masks, activations, embeddings, dropout, norms, residuals, and tied weights. |
| Training | Fluent Trainer/optimizer/loader APIs, MSE/cross-entropy, SGD/AdamW, groups/schedules, clipping, weighted evaluation, callbacks, stopping/cancellation, and deterministic batching. |
| Persistence | Named state, aliases, module configuration/modes, optimizer slots/counters, random counters, loader metadata, CRC validation, and atomic replacement. |
| Pretrained IO | Independent mapped GGUF v3/safetensors readers, checked local shard indexes, explicit weight names/transposes/ties, bounded tensor staging, and optional F16/BF16 storage expansion. |
| Transformers | Configurable multi-layer causal decoders, MHA/MQA/GQA, masks/RoPE, feed-forward variants, norms/tied readouts, and independent prefill/decode cache sessions. |
| Generation | Token-based single-sequence generation, Vectrix host sampling, top-k/top-p, penalties, stops, streaming, cancellation, and processed log probabilities. |
| Optional text | Independent tokenizer SPI, complete byte alphabet/BPE, GPT-2 files and a checked Hugging Face ByteLevel subset, special-token metadata, and incremental UTF-8 decoding. |

Tests include independent numerical references, gradient checks, nonlinear XOR
training, caller-defined transformer training, exact checkpoint resume, cached
attention parity, and a deferred backend for ordering/failure cleanup.

## Fluent model construction

See the [implemented NN/LM builder guide](docs/model-builders.md) for reusable
definitions, layer settings, definition/weight save-load, and cache sessions.
The [training guide](docs/training-guide.md) covers Trainer configuration, data,
evaluation, callbacks, and reproducible mid-epoch checkpoint resume. Explicit
training loops remain supported.
The [CNN guide](docs/cnn-guide.md) covers convolution/pooling geometry, gradients,
fluent builders, and the runnable image-classification example.
The [recurrent guide](docs/recurrent-guide.md) covers RNN/LSTM/GRU construction,
masked sequences, carried state, full/truncated backpropagation, and recall examples.
The [generation/tokenizer guide](docs/generation-guide.md) covers token sampling,
streaming text, reusable settings, dependency isolation, and remaining format support.
The [pretrained loading guide](docs/pretrained-loading.md) covers container ownership,
weight mappings, tokenizer assets, and supported format boundaries.
The [real pretrained-model guide](docs/pretrained-model-validation.md) covers the optional
Llama adapter, pinned SmolLM2 assets, reference generation and CPU acceptance evidence.

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
semantics and the adapters. Typed storage, FP32 arithmetic/broadcasts, batched GEMM,
activations, softmax, normalization, sums/norms, dropout and their supported gradients
have been exercised locally on Metal in both modes. Full NN/LLM inference and
Trainer workflows require further operation coverage; Vulkan execution is not
yet verified here. See the [implemented capability limits](docs/module-migration.md#initial-gpu-capabilities).
The [GPU neural-operation guide](docs/gpu-neural-operations.md) covers exact/throughput
selection, layout handling, gradient checks and remaining model-execution gates.

```sh
mvn -Pgpu,metal clean install
mvn -f examples/pom.xml -Pgpu verify
python3 tools/verify_module_consumers.py --gpu-metal
```

The [integration plan](docs/gpu-integration.md) records
ownership, distribution and the remaining gates, and
[numerical coverage](docs/numerical-coverage.md) maps every operation to its
kernels and lists the upstream requests. CPU users have no native GPU dependencies.

The merged [GGUF Metal experiment](nnjava-gguf-metal/README.md) separately supplies
a Llama 3.1 Q4_K_M reader, checkpoint tokenizer, and paged inference session. It
builds independently against private `1.15.0-step12-SNAPSHOT` GPU artifacts and
Vectrix 4.0.0; it is outside the main reactor/BOM. Its README records model-specific
accuracy and hardware limits. It does not change the pinned general-purpose
backend or imply complete GPU coverage through the tensor API.

## Plans and guides

- [CPU API guide](docs/cpu-guide.md)
- [Implemented NN/LM builders and cache sessions](docs/model-builders.md)
- [Fluent Trainer, evaluation, and checkpoint resume](docs/training-guide.md)
- [Convolution, pooling, and CNN examples](docs/cnn-guide.md)
- [RNN, LSTM, GRU, and recurrent state](docs/recurrent-guide.md)
- [Generation and optional byte-level tokenizers](docs/generation-guide.md)
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

Branched builders and broader
architecture examples remain planned in the roadmap. Mixed precision,
nested/higher-order differentiation, graph retention, distributed execution, and
further performance work remain later extensions. A production LLM and its
application behavior belong in the downstream model repository.

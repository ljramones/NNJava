# Fluent Neural Networks and Causal Language Models

These APIs are implemented on the Vectrix 4.1.0 CPU backend. They build ordinary
modules that work with existing explicit training loops, optimizers, state archives,
and checkpoint resume. The numerical stack remains pinned to DynamisGPU 1.14.1;
its NNJava adapter still needs additional operations before it can execute these
complete models on a GPU.

## Define and build a network

```java
var definition = NeuralNetwork.builder()
    .name("classifier")
    .inputFeatures(2)
    .seed(42)
    .dense("hidden", 8, Activation.Kind.TANH)
    .dropout("regularization", 0.1f)
    .dense("logits", 2)
    .define();

try (var context = ExecutionContext.cpu();
     var model = definition.build(context)) {
    // model.forward(input), gradient tapes, and optimizers use their usual APIs.
}
```

Builders are mutable and reusable. `define()` takes an immutable snapshot without
allocating tensors, probing hardware, or initializing numerical providers. A
definition can build multiple independent models. `build(context)` is the short
terminal form when a separate definition is unnecessary.

Supported sequential specifications include dense projections with optional
activations, grouped/depthwise convolution, max/average pooling, standalone
activations, dropout, LayerNorm/RMSNorm, flatten, and reshape. See the
[CNN guide](cnn-guide.md) for spatial settings. `Layers.dense(32).bias(false)` configures an immutable layer through
`.layer("projection", specification)`. `Linear.builder()` also supports explicit
input/output feature counts, bias selection, seed, definition, and construction.
Additional initializer families and precision policies remain planned.

Input shapes describe each sample's trailing axes. Leading batch/sequence axes
are preserved. For example, `.inputShape(2, 3).flatten("flat").dense("out", 4)`
maps `[batch, 2, 3]` to `[batch, 4]`. Inspect `definition.outputShape()` or infer a
complete result with `definition.outputShape(actualInputShape)` before allocation.

Named dense layers own a `linear` child, producing paths such as
`hidden.linear.weight`. Seeds derive from the root seed and stable component names;
inserting a parameterless layer does not reinitialize other named components.
Definitions validate shapes, duplicate names, provider operations, initialization,
and parameter allocation limits before constructing any layer.

## Construct a causal decoder

```java
var definition = CausalLanguageModel.builder()
    .name("application-decoder")
    .vocabularySize(64)
    .modelWidth(16)
    .blockCount(2)
    .contextLength(128)
    .attention(a -> a.queryHeads(4).keyValueHeads(2)
                     .rotaryPositions().rotaryBase(10_000).dropout(0.0f))
    .feedForward(f -> f.hiddenWidth(32).swiGlu())
    .normalization(NormalizationKind.RMS)
    .normalizationEpsilon(1e-5f)
    .tieInputOutputEmbeddings(true)
    .seed(42)
    .define();
```

The model accepts nonempty INT32 `[batch, time]` token IDs and returns FP32
`[batch, time, vocabulary]` logits. Labels, losses, vocabulary, tokenization, and
training recipes are supplied by the application. `forward(tokens, allowedMask)`
accepts a boolean mask broadcastable to `[batch, queryHeads, time, time]`; true
means allowed, and the causal mask is always applied.

Implemented options include MHA/MQA/GQA, attention projection biases, interleaved
RoPE with a configurable base or disabled positions, RMSNorm/LayerNorm,
SwiGLU or two-projection GELU/SiLU/ReLU feed-forward blocks, attention/residual/
embedding dropout, optional final normalization, and tied/separate readouts.
Readout and feed-forward projections are bias-free. Blocks use pre-normalization
and share a configuration template while owning distinct parameters and seeds.
Head divisibility and even rotary head dimensions are checked by `define()`.

`TransformerBlock` is also a public composable module with a validated immutable
`Config`. Custom architectures can continue to use the lower-level public modules.
Encoder/decoder models, MoE, rotary scaling, sliding windows, and other architecture
extensions remain in the [LLM roadmap](llm-roadmap.md).

## Persist definitions and weights

```java
ModelDefinitions.write(definitionPath, definition.specification());
var restoredDefinition = CausalLanguageModel.Definition.fromSpec(
    ModelDefinitions.read(definitionPath));

StateArchive.saveWeights(weightsPath, model);
try (var restored = restoredDefinition.build(context)) {
    StateArchive.read(weightsPath).restoreModel(restored);
}
```

`ComponentSpec` is a neutral tree in core. Its stable type IDs and schema versions
contain only configuration. Definition files have bounded size/depth/node counts,
a checksum, deterministic attribute ordering, and atomic replacement. Typed
decoders reject unknown versions or settings. No reflective class loading is used.
These files are separate from model weights and full training checkpoints.

Custom sequential layers implement `LayerDefinition`: shape inference, capability
validation, construction, and a specification. Supply explicit custom type decoders
to `NeuralNetwork.Definition.fromSpec(spec, decoderMap)`. Implementations must be
immutable and clean up their own allocations if their factory fails. Built-in
constructors and network composition clean up partial construction.

`StateArchive` remains independent of training. Full training resume uses
`training.Checkpoint`, or the [Trainer loop checkpoint API](training-guide.md);
decoder tests cover tied weights, optimizer slots, dropout
streams, mode, and identical subsequent updates after restore.

## Cached inference sessions

```java
model.training(false);
try (var session = model.newSession(batchSize);
     var prefixLogits = session.prefill(prefixTokenIds);
     var nextLogits = session.decode(nextTokenIds)) {
    // decode also accepts a chunk; session.length() tracks accumulated positions.
}
```

Each session owns one KV cache per block and borrows model parameters. Prefill
requires an empty session; decode requires a prefix. Calls validate batch size,
context capacity, and masks before cache mutation. Masks for decode broadcast to
`[batch, queryHeads, chunkLength, totalCachedLength]`.

Evaluation mode and absence of an active gradient tape are explicit requirements;
session creation never changes shared model mode. Parameter mutation generations
detect stale caches after optimizer updates or weight restore. Call `reset()` to
clear positions and adopt current weights before prefilling again. A compute
failure invalidates the session, which must be closed and recreated. A rejected
shape/mode/capacity check leaves it usable. Closing a session preserves model
weights; closing the model invalidates its sessions, whose caches still need closing.

This is the model-level cache API in transformer. A future generation module will
compose it with sampling, stopping, streaming, and tokenizer adapters. It introduces
no reverse dependency from transformer to generation.

## Runnable examples and checks

After `mvn clean install`:

```sh
mvn -f nnjava-examples/pom.xml exec:java -Dexec.args=mlp-xor
mvn -f nnjava-examples/pom.xml exec:java -Dexec.args=causal-lm
mvn -f examples/pom.xml verify
python3 tools/verify_module_consumers.py
```

The small examples train, save/reload definitions and weights, and verify their
results. The causal example also compares full and cached logits and reports
greedy next-token IDs. CPU smoke tests require no external corpus or weights.
The independent inference consumer uses capability modules without training/data
dependencies. Real pretrained-model loading and GPU model throughput are separate
milestones, not established by these synthetic learning examples.

## Verification record

The [2026-11-27 verification record](evidence/model-builders-2026-11-27.json) reports
95 passing tests across scalar/SIMD and Metal checks, including 17 decoder tests,
ten fluent-network tests, and two runnable example smoke tests. Model-only JPMS
consumption and the existing exact/throughput Metal consumers also pass.

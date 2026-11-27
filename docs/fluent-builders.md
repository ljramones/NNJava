# Fluent Model and Training Builders

Make common networks and transformer-based language models straightforward to
assemble while retaining custom modules and explicit runtime ownership. Builders
produce reusable configurations and definitions; a terminal build operation creates
the executable module in a chosen context.

Implemented now: sequential/Linear and causal-language-model definitions, named
NN builders, decoder configuration, custom-layer decoder hooks, checked definition
files, and CPU cache sessions. The [API guide](model-builders.md) documents exact
behavior and runnable examples. Generation and Trainer builders remain proposed.

## Builder layers

| Builder family                 | Creates                                                                                  | Module                                                                  |
|--------------------------------|------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| Layer builders                 | Validated Linear, Embedding, normalization, dropout, and other layer specifications.     | nn                                                                      |
| Network builders               | Sequential and, later, branched networks with named components and inferred shapes.      | nn                                                                      |
| Transformer builders           | Attention, feed-forward, block, encoder/decoder, and generic language-model definitions. | transformer                                                             |
| Generation builders            | Sampling/stopping definitions and inference sessions with explicit cache ownership.      | generation                                                              |
| Tokenizer and loading builders | Optional tokenizer providers, format readers, and explicit weight-name/layout mappings.  | tokenizers and io; provider-specific weight materialization in adapters |
| Training builders              | Optimizer/group, schedule, data-loader, evaluation, and Trainer configurations.          | training and data                                                       |
| Runtime builders               | Execution contexts with an explicit provider, precision policy, and random seed.         | core and backend providers                                              |

Put builders beside the functionality they configure. A convenience facade may
re-export common entry points; it should not become a separate implementation.

## Neural network example

```java
var definition = NeuralNetwork.builder()
    .name("digit-classifier")
    .inputFeatures(784)
    .seed(42)
    .dense("hidden", 256, Activation.Kind.RELU)
    .dropout("regularization", 0.1f)
    .dense("logits", 10)
    .define();

try (var model = definition.build(context)) {
    // Use ordinary forward calls, explicit training loops, and state archives.
}
```

A direct `.build(context)` terminal operation is also implemented. `.define()`
allocates no tensors or device resources and snapshots the reusable mutable builder.
Input sample shapes exclude leading batch/sequence axes; flatten and reshape
preserve those prefixes. Each dense component owns a `linear` child, so parameter
paths include `hidden.linear.weight`. Component seeds derive from the root seed
and stable layer name, independently of parameterless insertions.

`Layers.dense(units).bias(false)` and other immutable `LayerDefinition` values
configure advanced layers through `.layer(name, specification)`. The standalone
`Linear.builder().inputFeatures(2).outputFeatures(4).seed(42)` also supports
`.define()` and `.build(context)`.

Persist definitions independently of weights:

```java
ModelDefinitions.write(path, definition.specification());
var restored = NeuralNetwork.Definition.fromSpec(ModelDefinitions.read(path));
```

Files have a schema/header, checksum, bounded nesting/size, and atomic replacement.
Unknown component versions/settings are rejected by typed definition decoders.
Custom `LayerDefinition` implementations supply inference, validation, construction,
and a stable `ComponentSpec`; `fromSpec(spec, explicitDecoderMap)` restores their
registered type IDs. Custom factories own their construction-failure cleanup.

Initial layer specifications expose bias, activation, shape, and deterministic
seeds. Additional initializer families, precision overrides, and branch/merge
builders remain planned. Direct modules and custom layer definitions remain
available for application-specific composition.

## Classical architecture examples

The first-class [examples module](examples.md) exercises builders for CNNs,
vanilla RNNs, LSTMs, GRUs, and autoencoders as the supporting layers are added.
For example, the proposed CNN shape is:

```java
var imageClassifier = NeuralNetwork.builder()
    .inputShape(1, 28, 28)
    .conv2d("features", c -> c.filters(16).kernel(3).activation(ActivationKind.RELU))
    .maxPool2d("pool", 2)
    .flatten("flatten")
    .dense("logits", 10)
    .define();
```

Sequence builders should expose cell/state behavior and output selection:

```java
var forecaster = NeuralNetwork.builder()
    .sequenceInputFeatures(8)
    .lstm("memory", r -> r.hiddenSize(64).output(SequenceOutput.LAST))
    .dense("prediction", 1)
    .define();
```

Provide comparable RNN and GRU configuration, including state reset, masks,
sequence lengths, and optional truncated backpropagation. Keep architecture
examples in `nnjava-examples` and reusable convolution/recurrent layers in nn.

## Implemented language model builder

```java
var definition = CausalLanguageModel.builder()
    .name("application-decoder")
    .vocabularySize(32_000)
    .modelWidth(512)
    .blockCount(8)
    .contextLength(2_048)
    .attention(a -> a.queryHeads(8).keyValueHeads(2).rotaryPositions())
    .feedForward(f -> f.hiddenWidth(1_536).swiGlu())
    .normalization(NormalizationKind.RMS)
    .tieInputOutputEmbeddings(true)
    .seed(42)
    .define();

try (var model = definition.build(context)) {
    // Integer token IDs and masks produce logits; the application owns text policy.
}
```

A generic language-model composition belongs in the library. The consuming project
chooses architecture settings, weights, vocabulary/tokenizer configuration, corpus,
recipe, and application behavior. NNJava also supplies reusable generation,
optional tokenizers, weight readers, and adaptation mechanisms. The builder must
support customization of blocks, position handling, masking, weight sharing, and
feed-forward composition without requiring an existing named model family.

Build the language-model facade from the same public components applications use.
The implemented `model.newSession(batchSize)` owns per-block KV caches and checks
mode, context capacity, and stale weights. It borrows model-owned parameters. Tokenization and generation policy must not be mandatory
dependencies of model construction.

The [LLM roadmap](llm-roadmap.md) defines generation-session syntax, tokenizer and
weight-loading contracts, pretraining/SFT/LoRA builders, and their delivery order.
Generation settings are separate reusable definitions. A generation session borrows
model weights and owns its cache, sampling state, and temporary resources; it lives
in generation so transformer does not acquire a reverse dependency.

## Training builder proposal

```java
try (var trainer = Trainer.builder()
        .model(model)
        .optimizer(AdamW.builder().learningRate(0.001f).weightDecay(0.01f))
        .loss(CrossEntropy.builder().ignoreIndex(-100))
        .schedule(Schedules.warmupCosine(100, 10_000))
        .clipGlobalNorm(1.0)
        .checkpointEverySteps(500, checkpointDirectory)
        .build()) {
    trainer.fit(trainingData, validationData);
}
```

Retain explicit training loops as supported APIs. Trainer is an orchestration layer
for consistent batching, backward/update order, validation, metrics, callbacks,
checkpointing, early stopping, and cancellation. It must work with both ordinary
networks and caller-defined language models.

The proposed ownership rule is that Trainer borrows its model and input loaders,
and owns optimizer/work resources it creates from builder specifications. Attaching
an already-created optimizer requires an explicit ownership policy. No hidden
context, process-wide random generator, or background service is created.

## Validation and reproducibility requirements

- Check required settings, complete input/output shapes, and backend capabilities
  before allocating a partially built model. Errors identify the component path.
- Validate attention head divisibility, rotary dimensions, residual shapes,
  vocabulary/readout dimensions, weight ties, and context/cache settings together.
- Keep defaults documented and visible in the resulting definition. Reject
  contradictory settings; precision and backend changes are explicit.
- Derive reproducible initialization/dropout streams from a root seed and stable
  component identities. Definition-based construction and direct construction
  must agree under the same settings.
- Keep definitions immutable and reusable; avoid implicit mutation through nested
  builders. Decide and document whether builder instances themselves are reusable.
- Store stable component type IDs and schema versions with definitions. Support
  registration of custom component factories without embedding native handles.
- Preserve parameter aliases, module configuration, optimizer groups, random/data
  state, and execution mode through save/load and checkpoint resume.
- Close constructed modules and sessions explicitly. Roll back allocations when
  construction fails, and keep inference state separate from shared model weights.

## Builder acceptance examples

A user should be able to construct and train an XOR classifier with the short NN
builder, build a tiny decoder and compare it to direct component composition,
round-trip either definition before allocation, restore tied weights correctly,
and get a clear error for an invalid shape or unavailable numerical provider.
The same model definition should remain usable when a supported GPU context is
selected after the numerical stack is integrated. Expanded LLM acceptance also
covers loaded-model reference logits, growing caches, token streaming, masked
fine-tuning, and adapter save/load.

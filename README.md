# Neural Networks

A Java 25 deep-learning framework for building and training neural networks and
LLMs. The first implementation runs on the CPU through Vectrix. The planned
DynamisGPU adapter will use the same tensor operation boundary for Metal and Vulkan.

Applications define their models in separate repositories. This library supplies
tensors, automatic differentiation, composable layers, transformer primitives,
training utilities, and portable parameter persistence.

## Build and run

Requirements: JDK 25+, Maven, and the Vectrix artifact available through Maven.

```bash
mvn test                    # Numerical, gradient, ownership, and training tests
mvn clean verify            # Clean build, tests, and library JAR
mvn -q compile exec:java     # Train the public-API linear regression example
```

Coordinates: `org.dynamisengine:neuralnetworks:0.1.0-SNAPSHOT`. Sources use
`src/main/java/org/dynamisengine/nn`; tests use `src/test/java`.

Vectrix 3.0 remains the dependency target. The bootstrap build pins **2.6.0**
because the inspected checkout and local Maven cache still provide that version.
After 3.0 is released and available, run `mvn -Dvectrix.version=3.0.0 clean verify`
to validate the upgrade. The 3.0 artifact has not yet been tested here. Maven
resolves the dependency by coordinates, with no absolute source-tree dependency.

## Implemented first slice

- FP32, INT32, and BOOL storage; immutable shapes; checked sizes; row-major
  layouts; reshape, transpose, and read-only broadcast views.
- FP32 add/subtract/multiply/scale, total sum/mean, broadcast reduction, copying,
  and rank-two matrix multiplication through Vectrix GEMM.
- First-order reverse-mode differentiation with branching, shared-input
  accumulation, broadcast gradients, and saved-storage mutation protection.
- Named parameters, a composable `Module`/`UnaryModule` contract, `Linear`,
  mean-squared error, and SGD with deduplication of shared parameters.
- Backend-neutral operation requests and ordered completion handling, verified
  with a deferred backend that also exercises failure cleanup.

The regression example learns `y = 3x + 2`. It demonstrates library composition;
application-specific networks and LLM architectures belong in their own repos.

## API usage and ownership

```java
try (var context = ExecutionContext.cpu();
        var model = new Linear(context, 1, 1, 42);
        var optimizer = new Sgd(model.namedParameters().values(), 0.1f);
        var x = context.tensor(new float[] {-1, 0, 1}, Shape.of(3, 1));
        var y = context.tensor(new float[] {-1, 2, 5}, Shape.of(3, 1))) {
    for (int step = 0; step < 100; step++) {
        optimizer.zeroGrad();
        try (var tape = context.gradientTape();
                var prediction = model.forward(x);
                var loss = Losses.meanSquaredError(prediction, y)) {
            tape.backward(loss);
        }
        optimizer.step();
    }
}
```

See [LinearRegressionExample](src/main/java/org/dynamisengine/nn/examples/LinearRegressionExample.java)
for the complete imports and executable example.

Contexts are currently thread-confined and own backend resources. Close returned
tensor handles; `retain()` and `Parameter.value()` return additional owned handles.
Views, saved activations, and pending operations retain storage independently.
Host reads return copies and await pending work. Closing the context drains its
queue and invalidates remaining handles.

A gradient tape records operations in its scope, runs backward once, and releases
saved values. Gradients accumulate on trainable leaves until cleared. Operations
outside a tape run without recording; evaluation mode is independent of gradient
tracking. Updates run after backward. Internal `TensorAccess` methods are runtime
implementation details.

## Planning documents

- [Framework architecture](docs/architecture.md): scope, public abstractions,
  composition, and repository boundaries.
- [Tensor and backend contract](docs/backend-contract.md): operation semantics,
  memory ownership, execution, and CPU/GPU requirements.
- [DynamisGPU rework guidance](docs/dynamisgpu-rework.md): general compute APIs,
  native Metal integration, Vulkan changes, and migration steps.
- [DynamisGPU execution commission](docs/dynamisgpu-commission.md): implementation
  instructions and acceptance criteria for the session working in DynamisGPU.
- [Implementation roadmap](docs/roadmap.md): delivery stages and acceptance criteria.

## Remaining scope

The API is experimental. Batched matrix multiplication, axis reductions,
slicing/gather, activations, normalization, attention, additional optimizers,
checkpoints, and real GPU execution remain on the roadmap. Nested tapes,
higher-order gradients, graph retention, and implicit device transfers are not
implemented. Planning documents describe the larger target beyond this first slice.

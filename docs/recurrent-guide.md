# RNN, LSTM, and GRU networks

`nnjava-nn` supplies `RecurrentCell`, `Recurrent`, `RecurrentState`, and
`SequenceOutput`, plus fluent sequential layers. Numerical work composes existing
Vectrix-backed matrix, activation, selection, and tensor operations. There are no
NNJava recurrent numerical kernels or shaders. These APIs currently run on the
Vectrix 4.1.0 FP32 CPU path, in scalar and SIMD configurations.

## Build a sequence model

```java
var definition = NeuralNetwork.builder()
        .name("sequence-classifier").inputShape(8, 2).seed(42)
        .lstm("memory", 16, SequenceOutput.LAST)
        .dense("logits", 3)
        .define();
try (var model = definition.build(context)) {
    // [batch,8,2] -> [batch,3]
}
```

Imports come from `org.dynamisengine.nn.nn`. Comparable `.rnn(name, hidden)` and
`.gru(name, hidden)` methods are available. The default output is `ALL`:
`[batch,time,hidden]`; `LAST` returns `[batch,hidden]`. Dense layers act on the last
axis, so `ALL` followed by a dense head produces per-time predictions. Stack
recurrent layers by retaining `ALL` until the final sequence readout.

`Layers.rnn(hidden)`, `Layers.lstm(hidden)`, and `Layers.gru(hidden)` are immutable
settings supporting `.bias(false)` and `.output(SequenceOutput.LAST)` through
`.layer(name, specification)`. `define()` allocates no tensors; definitions use
versioned `ComponentSpec` schemas and round-trip with `ModelDefinitions`.
Network sample shapes fix `[time,features]`; standalone `Recurrent.builder()`
accepts different positive sequence lengths on successive calls.

The standalone builder specifies `kind`, `inputFeatures`, `hiddenSize`, `bias`,
`output`, and `seed`. Defaults are RNN, bias enabled, `ALL`, and seed 0; input and
hidden sizes are required. Cell builders have the same settings without output
selection. Both projections use the existing Linear normal initialization and
zero biases. Named component seeds remain stable across parameterless insertions.
Paths include `memory.cell.input.weight` and `memory.cell.recurrent.bias`.

## Cell equations and parameters

`RecurrentCell.Kind` selects an explicit convention:

- `RNN`: `h' = tanh(Wx*x + bx + Wh*h + bh)`.
- `LSTM`: gates are packed **input, forget, candidate, output (IFGO)**.
  `c' = sigmoid(f)*c + sigmoid(i)*tanh(g)` and
  `h' = sigmoid(o)*tanh(c')`. Both gate projections include their own bias.
- `GRU`: gates are **reset, update, candidate (RZN)**, using reset after the
  recurrent projection. With `a = Wx*x + bx` and `b = Wh*h + bh`,
  `r = sigmoid(a_r+b_r)`, `z = sigmoid(a_z+b_z)`,
  `n = tanh(a_n+r*b_n)`, and `h' = n+z*(h-n)`.
  In particular, the recurrent candidate bias is inside the reset multiplication.

For gate count G and hidden size H, projection weights are `[G*H,inputs]` and
`[G*H,H]`, with optional bias vectors `[G*H]`. LSTM's forget bias defaults to zero.
Cell `step(input,state)` borrows FP32 matrices `[batch,inputs]` and returns a new
owned state. `Recurrent.cell()` is a borrowed view of the registered cell for
step-by-step execution; the parent owns its lifetime.

## Explicit state and masked sequences

```java
try (var layer = Recurrent.builder().kind(RecurrentCell.Kind.LSTM)
                .inputFeatures(2).hiddenSize(16).seed(42).build(context);
        var initial = layer.zeroState(batchSize);
        var result = layer.run(input, initial, validLengths);
        var output = result.output();
        var finalState = result.state();
        var hidden = finalState.hidden();
        var cell = finalState.cell()) {
    // Each getter returns an additional owned handle: close it.
}
```

`run(input)` starts from zero; no call leaves hidden state inside the module.
`run(input, initial)` carries explicit values. Inputs have trailing `[time,features]`
axes; unbatched inputs and multiple leading batch axes are supported. Sequence
time and flattened batch size must be positive. State tensors are matrices
`[flattenedBatch,hidden]`, including `[1,hidden]` for an unbatched sequence.

Optional `validLengths` is INT32 with the input's batch-prefix shape, or a scalar
for unbatched input. Values lie in `[0,time]`. Lengths are right-padding lengths:
the layer zeroes padded inputs before cell computation, emits zero padded outputs,
and holds hidden/cell state after the last valid step. A zero-length row retains
its initial state; `LAST` returns that state. Length validation currently reads IDs
on the host. State selection and all numerical computation use provider operations.

Returned output/state are independent owned handles. Construct custom initial
state with `RecurrentState.hidden(h)` or `.lstm(h,c)`; these factories retain their
borrowed tensors. `resetRows(boolMask)` returns a new state with selected batch
rows zeroed. Like other Ops, resets preserve gradient history when performed
inside the active tape; computations outside a tape do not record history.

## Backpropagation and persistence

Unrolling inside one gradient tape provides full backpropagation through time,
including across multiple `run` calls that pass their state forward. Close old
handles as usual; the tape retains saved values independently. Create differentiable
sequence views inside the tape scope.

Use `state.detach()` at a truncated-BPTT boundary, or before carrying values into
a new tape. Detach retains values without their graph. Reusing undetached state
from another tape is rejected. Perform optimizer updates after backward and keep
the next chunk's detached state explicitly; detaching does not rewind values.

Model definitions, weights, and normal [Trainer checkpoints](training-guide.md)
use the existing APIs. The fluent model's `forward` starts each batch from zero
and treats every time position as valid. For variable lengths or carried state,
use `Recurrent.run` in an explicit loop or application-defined module. Downstream
layers can change padded zero outputs, so target/loss masking remains explicit.

Carried state is **application-owned** and is not automatically included in a
Trainer checkpoint. `state.snapshot()` returns neutral `TensorSnapshot` values;
`RecurrentState.fromSnapshot(context,snapshots)` restores detached state in a new
context. Persist these values together with the matching model/optimizer checkpoint,
stream position, and row identities when building stateful training. Snapshots
preserve values rather than autograd graphs. Model-only inference needs no training
or data-module dependency.

## Runnable examples and scope

```sh
mvn install
mvn -f nnjava-examples/pom.xml exec:java -Dexec.args=rnn
mvn -f nnjava-examples/pom.xml exec:java -Dexec.args=lstm
mvn -f nnjava-examples/pom.xml exec:java -Dexec.args=gru
```

These small delayed-recall classifiers remember the sign of an initial signal
through distractor steps, evaluate held-out samples, restore definitions, and
compare resumed/uninterrupted training. They are correctness fixtures rather than
long-memory or performance benchmarks. Tests additionally check independent cell
equations, every parameter's finite-difference gradient, masked/strided inputs,
chunked BPTT, detach/reset behavior, neutral state restore, constructor/forward
failure cleanup, and byte-identical final training archives.

GPU execution still requires more NNJava adapter coverage. Cells are currently
single-direction tanh RNN, standard LSTM, or reset-after GRU; projection LSTMs,
peepholes, bidirectional wrappers, recurrent dropout, and fused kernels remain
extensions. Stacking is ordinary layer composition. Unrolled tapes use memory
proportional to sequence length; recomputation and bounded-memory training are
separate work.

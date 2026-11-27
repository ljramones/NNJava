# Fluent training and reproducible resume

The `nnjava-training` and `nnjava-data` modules provide a supervised `Trainer`,
optimizer definitions, deterministic loaders, evaluation, callbacks, and complete
loop checkpoints. These APIs use the existing public tensor/autograd operations;
explicit training loops remain supported. Computation currently targets FP32 CPU
execution through Vectrix 4.1.0. The GPU adapter needs more operation coverage
before it can train these models.

## Configure a run

The example assumes a caller-owned `ExecutionContext`, `UnaryModule`, and
`Dataset` instances with `input` and `target` fields:

```java
var optimizer = Optimizers.adamW()
        .learningRate(.001f).weightDecay(.01f)
        .schedule(LearningRateSchedule.warmupCosine(100, 10_000, .1f))
        .define();
try (var training = DataLoader.builder(context, trainingDataset)
                .batchSize(32).shuffle(true).seed(42).build();
        var validation = DataLoader.builder(context, validationDataset)
                .batchSize(64).build();
        var trainer = Trainer.builder().model(model).optimizer(optimizer)
                .objective(Objectives.crossEntropy(-100))
                .fields("input", "target").clipGlobalNorm(1)
                .earlyStopping(3, .001)
                .checkpointEvery(500, Path.of("training.nnj"))
                .callback(event -> {
                    if (event.kind() == Trainer.EventKind.EPOCH) {
                        System.out.println(event.evaluation().loss());
                    }
                    return Trainer.Decision.CONTINUE;
                }).build()) {
    var result = trainer.fit(training, validation, 20);
    var evaluation = trainer.evaluate(validation);
    trainer.save(Path.of("final.nnj"), training);
}
```

Imports come from `org.dynamisengine.nn.data`,
`org.dynamisengine.nn.training`, and `java.nio.file.Path`. Both ordinary networks
and `CausalLanguageModel` implement the same `UnaryModule` boundary. Language-model
datasets supply already shifted INT32 inputs/targets shaped `[batch, time]`;
the application chooses tokenization, document boundaries, and masking policy.
`fields("tokens", "labels")` selects alternative batch names.

Optimizer defaults are AdamW rate `.001`, betas `.9/.999`, epsilon `1e-8`,
zero decay, and a constant schedule. `Optimizers.sgd()` defaults to rate `.01`,
zero momentum/decay, and no Nesterov acceleration. `define()` snapshots the
builder; no slots are allocated until binding parameters. Existing `ParameterGroup`
and schedule APIs remain available through an `OptimizerFactory` lambda, for
example `parameters -> new AdamW(groups(parameters), .9f, .999f, 1e-8f, schedule)`.
A custom factory must return a fresh owned optimizer for the supplied parameters.

Loader defaults are batch size 32, no shuffle, no dropped final batch, and seed 0.
Builders are reusable; each loader has its own cursor and deterministic epoch
ordering. A loader borrows its dataset; `TensorDataset` retains detached tensor
handles. Closing a Trainer releases its optimizer; it borrows the model, loaders,
context, and callbacks. Close resources in reverse construction order.

## Evaluation, callbacks, and stopping

`fit(loader, epochs)` completes additional epochs; a resumed partial epoch is the
first of those epochs. `fit(loader, validation, epochs, maximumBatches)` also
bounds work for interruption or interactive runs. A completed epoch advances the
training loader to the next epoch's start. Progress distinguishes committed
**batches**, optimizer **steps**, and **completedEpochs**.

Losses aggregate by their true denominator: MSE counts target elements and cross
entropy counts targets other than `ignoreIndex`. A zero-weight batch advances
the loader without a forward/backward pass or optimizer/schedule update. Empty
evaluations return loss 0, weight 0, and score NaN. Cross entropy reports accuracy;
MSE has no score. A custom `TrainingObjective` returns an owned scalar mean loss,
its denominator, an optional mean score, and a stable configuration ID. Its score
must use the same denominator. Weight/score calculation must have no random or
model-state side effects. The current cross-entropy denominator reads target IDs
on the host; future resident GPU training will need a reduction-based path.

Evaluation covers the entire current loader epoch, respects `dropLast`, and
restores its cursor. Training and evaluation restore caller-selected module modes,
including mixed child modes, on success or failure. Evaluation neither records
gradients nor advances dropout streams.

Callbacks run synchronously after a committed batch. Event order is STEP,
then EPOCH and VALIDATION at epoch boundaries. Return `Decision.STOP` or call
`requestStop()` to stop before another batch. Only `requestStop()` may be called
from another thread. Callbacks must not mutate the borrowed model/loader or
reenter/close the Trainer; callback state is application-owned and is not saved.
An exception in a callback leaves the completed batch committed.

Early stopping requires validation and minimizes validation loss. Improvement
must exceed `minimumDelta`; `patience` counts consecutive non-improving validation
epochs. The stopping state is checkpointed. A restored stopped run stays stopped;
this API does not silently restore earlier best weights. Save desired best-model
snapshots separately. Periodic checkpoints count batches, including zero-weight
batches, and use atomic replacement of one file.

## Save and resume

Build the same model definition, optimizer/schedule, objective, field mapping,
clipping/stopping settings, and loader configuration, then call:

```java
trainer.restore(Path.of("training.nnj"), training);
trainer.fit(training, validation, remainingEpochs);
```

The checkpoint includes weights/aliases, module modes and random counters,
optimizer slots/counters/configuration, shuffled loader position, partial-epoch
metric totals, loop counters, and early-stopping state. Automatic checkpoints
preserve caller modes rather than persisting the Trainer's temporary training
mode. Model, optimizer, loop, and loader compatibility are checked before restore
changes live state. Changing callback instances or checkpoint frequency/path is
allowed. Dataset contents and validation data are application responsibilities;
size/configuration checks do not fingerprint a corpus. Record corpus identity in
application run metadata when reproducibility depends on it.

The training loader must start at an epoch boundary or be restored through
Trainer. Do not move its cursor outside Trainer between calls. A pre-update
forward/loss/backward failure rolls back its cursor and module random state so a
corrected batch can be retried. An optimizer failure may have changed parameters;
Trainer rejects further training, evaluation, and save until a known checkpoint
is restored. The same rule applies if epoch bookkeeping/validation fails after
an update, so incomplete stopping/metric state cannot be silently continued.
Context/backend failures can additionally require recreating the
context. This is not a transactional rollback of arbitrary custom module side
effects or optimizer updates.

Run the complete autoencoder workflow with:

```sh
mvn install
mvn -f nnjava-examples/pom.xml exec:java -Dexec.args=autoencoder
```

The example learns a two-dimensional bottleneck, evaluates reconstruction, and
compares resumed and uninterrupted training. Its tiny synthetic data is a
correctness demonstration, not evidence of real-model training throughput.

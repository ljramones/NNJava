package org.dynamisengine.nn.training;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import org.dynamisengine.nn.data.DataLoader;
import org.dynamisengine.nn.nn.Module;
import org.dynamisengine.nn.nn.UnaryModule;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Tensor;

/**
 * Thread-confined supervised training. Borrows its model/loaders and owns its optimizer.
 * Explicit training loops remain supported independently of this orchestration API.
 */
public final class Trainer implements AutoCloseable {
    private final ExecutionContext context;
    private final UnaryModule model;
    private final Optimizer optimizer;
    private final TrainingObjective objective;
    private final String inputField, targetField;
    private final double clipNorm, minimumDelta;
    private final int patience;
    private final long checkpointInterval;
    private final Path checkpointPath;
    private final List<TrainingCallback> callbacks;
    private final AtomicBoolean stop = new AtomicBoolean();
    private long epochs, steps, batches, badEpochs;
    private double bestLoss = Double.POSITIVE_INFINITY;
    private final Totals epochTotals = new Totals();
    private DataLoader.State expectedLoader;
    private boolean running, closed, requiresRestore;

    private Trainer(Builder builder) {
        model = Objects.requireNonNull(builder.model, "model");
        objective = Objects.requireNonNull(builder.objective, "objective");
        if (objective.id() == null || objective.id().isBlank()) {
            throw new IllegalArgumentException("Objective needs a stable configuration identity");
        }
        inputField = field(builder.inputField); targetField = field(builder.targetField);
        clipNorm = builder.clipNorm; patience = builder.patience; minimumDelta = builder.minimumDelta;
        if (!Double.isFinite(clipNorm) || clipNorm < 0 || patience < 0 || !Double.isFinite(minimumDelta) || minimumDelta < 0
                || builder.checkpointInterval < 0 || builder.checkpointInterval > 0 && builder.checkpointPath == null) {
            throw new IllegalArgumentException("Invalid clipping, stopping, or checkpoint configuration");
        }
        callbacks = List.copyOf(builder.callbacks);
        checkpointInterval = builder.checkpointInterval; checkpointPath = builder.checkpointPath;
        var parameters = model.namedParameters();
        if (parameters.isEmpty()) { throw new IllegalArgumentException("Trainer needs model parameters"); }
        try (var value = parameters.values().iterator().next().value()) { context = value.context(); }
        context.ensureOpen();
        if (context.activeTape() != null) { throw new IllegalStateException("Build Trainer outside a gradient tape"); }
        for (var parameter : parameters.values()) {
            try (var value = parameter.value()) { context.requireContext(value); }
        }
        optimizer = Objects.requireNonNull(Objects.requireNonNull(builder.optimizer, "optimizer").build(parameters));
        try {
            if (optimizer.steps() != 0) { throw new IllegalArgumentException("Optimizer factory must create fresh state"); }
        } catch (RuntimeException error) { optimizer.close(); throw error; }
    }

    private static String field(String value) {
        if (value == null || value.isBlank()) { throw new IllegalArgumentException("Nonblank batch field required"); }
        return value;
    }

    public static Builder builder() { return new Builder(); }

    public Progress progress() { ensureOpen(); return new Progress(epochs, steps, batches); }

    /** May be called from another thread; cancellation is observed between complete batches. */
    public void requestStop() { stop.set(true); }

    public Result fit(DataLoader training, int additionalEpochs) throws IOException {
        return fit(training, null, additionalEpochs, Long.MAX_VALUE);
    }

    public Result fit(DataLoader training, DataLoader validation, int additionalEpochs) throws IOException {
        return fit(training, validation, additionalEpochs, Long.MAX_VALUE);
    }

    /** A partially completed epoch counts as the first requested epoch when resuming. */
    public Result fit(DataLoader training, DataLoader validation, int additionalEpochs, long maximumBatches) throws IOException {
        ensureIdle();
        if (requiresRestore) { throw new IllegalStateException("Training state is incomplete; restore a checkpoint before training"); }
        if (additionalEpochs < 1 || maximumBatches < 1) { throw new IllegalArgumentException("Positive epoch and batch limits required"); }
        checkLoader(training);
        if (validation != null) { checkLoader(validation); }
        if (patience > 0 && validation == null) { throw new IllegalArgumentException("Early stopping requires validation data"); }
        DataLoader.State initial = training.state();
        if (expectedLoader == null && initial.cursor() != 0) { throw new IllegalArgumentException("Start at an epoch boundary or restore a Trainer checkpoint"); }
        if (expectedLoader != null && !expectedLoader.equals(initial)) { throw new IllegalArgumentException("Training loader changed outside Trainer"); }
        if (!training.hasNext()) { throw new IllegalArgumentException("Training loader has no batches"); }
        if (patience > 0 && badEpochs >= patience) {
            return new Result(progress(), epochTotals.result(), null, StopReason.EARLY_STOPPING);
        }
        long endEpoch = Math.addExact(epochs, additionalEpochs), endBatch = Math.addExact(batches, Math.min(maximumBatches, Long.MAX_VALUE - batches));
        var modes = modes();
        Evaluation trainingResult = epochTotals.result(), validationResult = null;
        StopReason reason = StopReason.EPOCH_LIMIT;
        expectedLoader = initial;
        running = true;
        try {
            model.training(true);
            while (epochs < endEpoch && batches < endBatch) {
                if (stop.get()) { reason = StopReason.REQUESTED; break; }
                var before = training.state();
                var randomState = model.trainingState();
                Sample sample;
                try (var batch = training.next(); var input = batch.get(inputField); var target = batch.get(targetField)) {
                    sample = trainBatch(input, target);
                } catch (RuntimeException error) {
                    if (!requiresRestore) {
                        training.restore(before);
                        model.restoreTrainingState(randomState);
                        optimizer.zeroGrad();
                    }
                    throw error;
                }
                requiresRestore = true;
                batches = Math.addExact(batches, 1);
                epochTotals.add(sample);
                boolean finishedEpoch = !training.hasNext();
                if (finishedEpoch) {
                    epochs = Math.addExact(epochs, 1);
                    trainingResult = epochTotals.result();
                    epochTotals.clear();
                    training.nextEpoch();
                    expectedLoader = training.state();
                    if (validation != null) { validationResult = evaluateInternal(validation); }
                    if (patience > 0 && validationResult.weight() > 0) {
                        if (validationResult.loss() < bestLoss - minimumDelta) {
                            bestLoss = validationResult.loss(); badEpochs = 0;
                        } else { badEpochs = Math.addExact(badEpochs, 1); }
                    }
                } else { trainingResult = epochTotals.result(); }
                expectedLoader = training.state();
                requiresRestore = false;
                emit(EventKind.STEP, new Evaluation(sample.loss, sample.score, sample.weight, 1));
                if (finishedEpoch) {
                    emit(EventKind.EPOCH, trainingResult);
                    if (validationResult != null) { emit(EventKind.VALIDATION, validationResult); }
                }
                if (checkpointInterval > 0 && batches % checkpointInterval == 0) {
                    var trainingModes = modes();
                    try { restoreModes(modes); saveInternal(checkpointPath, training); }
                    finally { restoreModes(trainingModes); }
                }
                if (patience > 0 && badEpochs >= patience) { reason = StopReason.EARLY_STOPPING; break; }
                if (stop.get()) { reason = StopReason.REQUESTED; break; }
                if (batches == endBatch && epochs < endEpoch) { reason = StopReason.BATCH_LIMIT; }
            }
            return new Result(progress(), trainingResult, validationResult, reason);
        } finally {
            try { restoreModes(modes); } finally { running = false; stop.set(false); }
        }
    }

    private Sample trainBatch(Tensor input, Tensor target) {
        context.requireContext(input); context.requireContext(target);
        optimizer.zeroGrad();
        long weight = weight(target);
        if (weight == 0) { return new Sample(0, Double.NaN, 0); }
        Sample sample;
        try (var tape = context.gradientTape(); var prediction = model.forward(input); var loss = objective.loss(prediction, target)) {
            sample = sample(prediction, target, loss, weight);
            tape.backward(loss);
        }
        if (clipNorm > 0) { Gradients.clipGlobalNorm(model.namedParameters().values(), clipNorm); }
        requiresRestore = true;
        optimizer.step();
        steps = Math.addExact(steps, 1);
        requiresRestore = false;
        return sample;
    }

    /** Evaluates the complete loader epoch without consuming its cursor or RNG state. */
    public Evaluation evaluate(DataLoader loader) {
        ensureIdle();
        if (requiresRestore) { throw new IllegalStateException("Restore incomplete training state first"); }
        running = true;
        try { return evaluateInternal(loader); } finally { running = false; }
    }

    private Evaluation evaluateInternal(DataLoader loader) {
        checkLoader(loader);
        var state = loader.state();
        var modes = modes();
        var totals = new Totals();
        try {
            model.training(false);
            loader.restore(new DataLoader.State(state.seed(), state.epoch(), 0, state.batchSize(), state.shuffle(), state.dropLast(), state.size()));
            while (loader.hasNext()) {
                try (var batch = loader.next(); var input = batch.get(inputField); var target = batch.get(targetField)) {
                    context.requireContext(input); context.requireContext(target);
                    long weight = weight(target);
                    if (weight == 0) { totals.add(new Sample(0, Double.NaN, 0)); continue; }
                    try (var prediction = model.forward(input); var loss = objective.loss(prediction, target)) {
                        totals.add(sample(prediction, target, loss, weight));
                    }
                }
            }
            return totals.result();
        } finally {
            try { loader.restore(state); } finally { restoreModes(modes); }
        }
    }

    private long weight(Tensor target) {
        long value = objective.weight(target);
        if (value < 0) { throw new IllegalArgumentException("Objective weight cannot be negative"); }
        return value;
    }

    private Sample sample(Tensor prediction, Tensor target, Tensor loss, long weight) {
        context.requireContext(loss);
        if (loss.shape().rank() != 0) { throw new IllegalArgumentException("Objective must return a scalar mean loss"); }
        double value = loss.item(), score = objective.score(prediction, target);
        if (!Double.isFinite(value) || !Double.isNaN(score) && !Double.isFinite(score)) {
            throw new ArithmeticException("Nonfinite objective loss or score");
        }
        return new Sample(value, score, weight);
    }

    private void emit(EventKind kind, Evaluation evaluation) {
        var event = new Event(kind, progress(), evaluation);
        for (var callback : callbacks) {
            if (Objects.requireNonNull(callback.onEvent(event)) == Decision.STOP) { stop.set(true); }
        }
    }

    public void save(Path path, DataLoader training) throws IOException {
        ensureIdle();
        if (requiresRestore) { throw new IllegalStateException("Cannot save incomplete training state"); }
        saveInternal(path, training);
    }

    private void saveInternal(Path path, DataLoader training) throws IOException {
        checkLoader(training);
        var state = training.state();
        if (expectedLoader == null && state.cursor() != 0 || expectedLoader != null && !expectedLoader.equals(state)) {
            throw new IllegalArgumentException("Training loader state does not match Trainer");
        }
        var metadata = new LinkedHashMap<>(configuration());
        metadata.putAll(state.metadata());
        metadata.put("trainer.epochs", Long.toString(epochs)); metadata.put("trainer.steps", Long.toString(steps));
        metadata.put("trainer.batches", Long.toString(batches)); metadata.put("trainer.bestLoss", Double.toHexString(bestLoss));
        metadata.put("trainer.badEpochs", Long.toString(badEpochs));
        metadata.put("trainer.lossSum", Double.toHexString(epochTotals.loss));
        metadata.put("trainer.scoreSum", Double.toHexString(epochTotals.score));
        metadata.put("trainer.weight", Long.toString(epochTotals.weight)); metadata.put("trainer.epochBatches", Long.toString(epochTotals.batches));
        Checkpoint.save(path, model, optimizer, metadata);
    }

    /** Checks all model, optimizer, loader, and loop configuration before applying state. */
    public void restore(Path path, DataLoader training) throws IOException {
        ensureIdle(); checkLoader(training);
        var loaded = Checkpoint.read(path);
        var metadata = loaded.metadata();
        configuration().forEach((key, value) -> {
            if (!value.equals(metadata.get(key))) { throw new IllegalArgumentException("Trainer configuration mismatch: " + key); }
        });
        var state = DataLoader.State.fromMetadata(metadata);
        training.validateState(state);
        long restoredEpochs = counter(metadata, "epochs"), restoredSteps = counter(metadata, "steps"), restoredBatches = counter(metadata, "batches");
        long restoredBad = counter(metadata, "badEpochs"), weight = counter(metadata, "weight"), epochBatches = counter(metadata, "epochBatches");
        double restoredBest = Double.parseDouble(metadata.get("trainer.bestLoss"));
        double loss = Double.parseDouble(metadata.get("trainer.lossSum")), score = Double.parseDouble(metadata.get("trainer.scoreSum"));
        if (!Double.isFinite(loss) || !Double.isNaN(score) && !Double.isFinite(score) || Double.isNaN(restoredBest)
                || restoredBest == Double.NEGATIVE_INFINITY || restoredSteps > restoredBatches || restoredEpochs > restoredBatches
                || epochBatches > restoredBatches || epochBatches == 0 && (weight != 0 || loss != 0 || score != 0)
                || epochBatches != state.cursor() / state.batchSize()
                || loaded.optimizer() == null || !Objects.equals(loaded.optimizer().counters().get("steps"), restoredSteps)) {
            throw new IllegalArgumentException("Invalid Trainer checkpoint counters or totals");
        }
        loaded.restore(model, optimizer);
        training.restore(state);
        epochs = restoredEpochs; steps = restoredSteps; batches = restoredBatches; badEpochs = restoredBad; bestLoss = restoredBest;
        epochTotals.loss = loss; epochTotals.score = score; epochTotals.weight = weight; epochTotals.batches = epochBatches;
        expectedLoader = state; requiresRestore = false;
    }

    private static long counter(Map<String, String> metadata, String key) {
        long value = Long.parseLong(metadata.get("trainer." + key));
        if (value < 0) { throw new IllegalArgumentException("Negative Trainer counter " + key); }
        return value;
    }

    private Map<String, String> configuration() {
        return Map.of("trainer.version", "1", "trainer.objective", objective.id(), "trainer.input", inputField, "trainer.target", targetField,
                "trainer.clipNorm", Double.toHexString(clipNorm), "trainer.patience", Integer.toString(patience), "trainer.minimumDelta", Double.toHexString(minimumDelta));
    }

    private void checkLoader(DataLoader loader) {
        if (Objects.requireNonNull(loader).context() != context) { throw new IllegalArgumentException("Loader and model contexts differ"); }
    }

    private List<Mode> modes() {
        var result = new ArrayList<Mode>();
        collectModes(model, new IdentityHashMap<>(), result);
        return result;
    }

    private static void collectModes(Module module, IdentityHashMap<Module, Boolean> seen, List<Mode> modes) {
        if (seen.put(module, Boolean.TRUE) == null) {
            modes.add(new Mode(module, module.isTraining()));
            module.children().values().forEach(child -> collectModes(child, seen, modes));
        }
    }

    private static void restoreModes(List<Mode> modes) { modes.forEach(mode -> mode.module.training(mode.training)); }

    private void ensureOpen() {
        context.ensureOpen();
        if (closed) { throw new IllegalStateException("Trainer is closed"); }
    }

    private void ensureIdle() {
        ensureOpen();
        if (running || context.activeTape() != null) { throw new IllegalStateException("Trainer requires an idle context outside a tape"); }
    }

    @Override public void close() {
        context.checkThread();
        if (!closed) {
            if (running) { throw new IllegalStateException("Cannot close a running Trainer; requestStop instead"); }
            optimizer.close(); closed = true;
        }
    }

    public enum Decision { CONTINUE, STOP }
    public enum EventKind { STEP, EPOCH, VALIDATION }
    public enum StopReason { EPOCH_LIMIT, BATCH_LIMIT, REQUESTED, EARLY_STOPPING }
    public record Progress(long completedEpochs, long steps, long batches) { }
    /** Score is accuracy for cross entropy and NaN when an objective supplies no score. */
    public record Evaluation(double loss, double score, long weight, long batches) { }
    public record Event(EventKind kind, Progress progress, Evaluation evaluation) { }
    public record Result(Progress progress, Evaluation training, Evaluation validation, StopReason reason) { }
    private record Mode(Module module, boolean training) { }
    private record Sample(double loss, double score, long weight) { }

    private static final class Totals {
        double loss, score;
        long weight, batches;
        void add(Sample sample) {
            batches = Math.addExact(batches, 1);
            if (sample.weight == 0) { return; }
            loss += sample.loss * sample.weight; score += sample.score * sample.weight;
            weight = Math.addExact(weight, sample.weight);
        }
        Evaluation result() { return new Evaluation(weight == 0 ? 0 : loss / weight, weight == 0 ? Double.NaN : score / weight, weight, batches); }
        void clear() { loss = 0; score = 0; weight = 0; batches = 0; }
    }

    public static final class Builder {
        private UnaryModule model;
        private OptimizerFactory optimizer = Optimizers.adamW().define();
        private TrainingObjective objective = Objectives.crossEntropy();
        private String inputField = "input", targetField = "target";
        private double clipNorm, minimumDelta;
        private int patience;
        private long checkpointInterval;
        private Path checkpointPath;
        private final List<TrainingCallback> callbacks = new ArrayList<>();
        private Builder() { }
        public Builder model(UnaryModule value) { model = value; return this; }
        public Builder optimizer(OptimizerFactory value) { optimizer = value; return this; }
        public Builder objective(TrainingObjective value) { objective = value; return this; }
        public Builder fields(String input, String target) { inputField = input; targetField = target; return this; }
        public Builder clipGlobalNorm(double maximum) {
            if (!Double.isFinite(maximum) || maximum <= 0) { throw new IllegalArgumentException("Positive finite clipping norm required"); }
            clipNorm = maximum; return this;
        }
        public Builder earlyStopping(int epochsWithoutImprovement, double delta) {
            if (epochsWithoutImprovement < 1) { throw new IllegalArgumentException("Positive patience required"); }
            patience = epochsWithoutImprovement; minimumDelta = delta; return this;
        }
        /** Writes after complete batches, including callback/validation/stopping state. */
        public Builder checkpointEvery(long batchInterval, Path path) {
            if (batchInterval < 1) { throw new IllegalArgumentException("Positive checkpoint interval required"); }
            checkpointInterval = batchInterval; checkpointPath = Objects.requireNonNull(path); return this;
        }
        public Builder callback(TrainingCallback value) { callbacks.add(Objects.requireNonNull(value)); return this; }
        public Trainer build() { return new Trainer(this); }
    }
}

package org.dynamisengine.nn;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import org.dynamisengine.nn.data.DataLoader;
import org.dynamisengine.nn.data.TensorDataset;
import org.dynamisengine.nn.io.StateDict;
import org.dynamisengine.nn.nn.Activation;
import org.dynamisengine.nn.nn.Linear;
import org.dynamisengine.nn.nn.NeuralNetwork;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;
import org.dynamisengine.nn.training.*;
import org.dynamisengine.nn.transformer.CausalLanguageModel;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class TrainerTest {
    @TempDir Path temporary;

    private static NeuralNetwork.Definition network() {
        return NeuralNetwork.builder().name("training-fixture").inputFeatures(2).seed(42)
                .dense("hidden", 5, Activation.Kind.TANH).dropout("dropout", .2f).dense("output", 2).define();
    }

    private static TensorDataset dataset(ExecutionContext context) {
        try (var input = context.tensor(new float[]{0,0,0,1,1,0,1,1,.5f,.25f}, Shape.of(5,2));
             var target = context.tensor(new int[]{0,1,1,0,1}, Shape.of(5))) {
            return new TensorDataset(Map.of("input", input, "target", target));
        }
    }

    private static DataLoader loader(ExecutionContext context, TensorDataset dataset) {
        return DataLoader.builder(context, dataset).batchSize(2).shuffle(true).seed(123).build();
    }

    private static OptimizerFactory adam() {
        return Optimizers.adamW().learningRate(.025f).weightDecay(.01f)
                .schedule(LearningRateSchedule.warmupCosine(2, 30, .1f)).define();
    }

    private static Trainer trainer(NeuralNetwork model) {
        return Trainer.builder().model(model).optimizer(adam()).clipGlobalNorm(.4).build();
    }

    private static void sameWeights(org.dynamisengine.nn.nn.Module first, org.dynamisengine.nn.nn.Module second) {
        assertEquals(first.namedParameters().keySet(), second.namedParameters().keySet());
        first.namedParameters().forEach((name, parameter) -> {
            try (var a = parameter.value(); var b = second.namedParameters().get(name).value()) {
                assertArrayEquals(a.toFloatArray(), b.toFloatArray(), name);
            }
        });
        assertEquals(first.trainingState(), second.trainingState());
    }

    @Test void optimizerDefinitionsSnapshotBuilderOptionsAndValidateBeforeAllocation() {
        var builder = Optimizers.adamW().learningRate(.02f).betas(.8f,.95f).epsilon(1e-6f);
        var definition = builder.define(); builder.learningRate(.4f);
        assertEquals(.02f, definition.learningRate());
        assertThrows(IllegalArgumentException.class, () -> Optimizers.adamW().betas(1,.9f).define());
        assertThrows(IllegalArgumentException.class, () -> Optimizers.sgd().nesterov(true).define());
        assertThrows(IllegalArgumentException.class, () -> Optimizers.sgd().learningRate(Float.NaN).define());
        try (var context = ExecutionContext.cpu(); var model = new Linear(context,2,2,42)) {
            int before = context.liveAllocations();
            try (var optimizer = definition.build(model.namedParameters())) {
                assertEquals(0, optimizer.steps()); assertTrue(context.liveAllocations() > before);
            }
            assertEquals(before, context.liveAllocations());
        }
    }

    @Test void loaderBuildersHaveIndependentDeterministicCursorsAndBorrowTheirDataset() {
        try (var context = ExecutionContext.cpu(); var data = dataset(context)) {
            var builder = DataLoader.builder(context,data).batchSize(2).shuffle(true).seed(123);
            try (var first = builder.build(); var second = builder.build(); var batch = first.next()) {
                assertEquals(0,second.state().cursor()); assertEquals(2,first.state().cursor());
                var state = first.state(); second.restore(state); assertEquals(state,second.state());
                var metadata = new LinkedHashMap<>(state.metadata()); metadata.put("loader.shuffle","yes");
                assertThrows(IllegalArgumentException.class, () -> DataLoader.State.fromMetadata(metadata));
                assertThrows(IllegalArgumentException.class, () -> second.validateState(new DataLoader.State(7,0,0,2,true,false,5)));
                assertEquals(state,second.state());
            }
            assertEquals(5,data.size());
        }
    }

    @Test void trainerMatchesAnExplicitLoopWithDropoutClippingAndScheduledAdam() throws Exception {
        try (var context = ExecutionContext.cpu(); var data = dataset(context); var a = network().build(context); var b = network().build(context);
             var first = loader(context,data); var second = loader(context,data); var trainer = trainer(a); var optimizer = adam().build(b.namedParameters())) {
            var result = trainer.fit(first,3);
            for (int epoch=0; epoch<3; epoch++) {
                while (second.hasNext()) {
                    optimizer.zeroGrad();
                    try (var batch = second.next(); var input = batch.get("input"); var target = batch.get("target");
                         var tape = context.gradientTape(); var output = b.forward(input); var loss = Losses.crossEntropy(output,target)) { tape.backward(loss); }
                    Gradients.clipGlobalNorm(b.namedParameters().values(),.4); optimizer.step();
                }
                second.nextEpoch();
            }
            sameWeights(a,b); assertEquals(first.state(),second.state());
            assertEquals(new Trainer.Progress(3,9,9),result.progress());
            assertEquals(Trainer.StopReason.EPOCH_LIMIT,result.reason());
        }
    }

    @Test void evaluationWeightsUnevenBatchesAndPreservesLoaderAndMixedModes() {
        try (var context = ExecutionContext.cpu(); var model = NeuralNetwork.builder().inputFeatures(1).dense("out",1).build(context);
             var input = context.tensor(new float[5],Shape.of(5,1)); var target = context.tensor(new float[]{0,0,0,0,10},Shape.of(5,1));
             var data = new TensorDataset(Map.of("input",input,"target",target)); var loader = DataLoader.builder(context,data).batchSize(4).build();
             var trainer = Trainer.builder().model(model).objective(Objectives.meanSquaredError()).build()) {
            for (var parameter : model.namedParameters().values()) { try (var value = parameter.value(); var zero = context.zeros(value.dtype(),value.shape())) { context.assign(value,zero); } }
            model.training(false); model.children().get("out").training(true);
            var modes = model.trainingState();
            try (var batch = loader.next()) { /* Evaluate from the beginning despite this consumed batch. */ }
            var state = loader.state(); var result = trainer.evaluate(loader);
            assertEquals(20,result.loss(),1e-6); assertEquals(5,result.weight()); assertEquals(2,result.batches());
            assertTrue(Double.isNaN(result.score())); assertEquals(state,loader.state()); assertEquals(modes,model.trainingState());
        }
    }

    @Test void tokenEvaluationUsesValidTargetCountsRatherThanBatchSizes() {
        try (var context = ExecutionContext.cpu(); var model = new Linear(context,1,2,42);
             var input = context.tensor(new float[]{1,-1,2,1,-2},Shape.of(5,1)); var target = context.tensor(new int[]{0,-100,1,0,1},Shape.of(5));
             var data = new TensorDataset(Map.of("input",input,"target",target)); var loader = DataLoader.builder(context,data).batchSize(3).build();
             var trainer = Trainer.builder().model(model).build()) {
            try (var weights = model.namedParameters().get("weight").value(); var chosen = context.tensor(new float[]{1,-1},weights.shape())) { context.assign(weights,chosen); }
            try (var bias = model.namedParameters().get("bias").value(); var zero = context.zeros(bias.dtype(),bias.shape())) { context.assign(bias,zero); }
            try (var prediction = model.forward(input); var loss = Losses.crossEntropy(prediction,target)) {
                var result = trainer.evaluate(loader);
                assertEquals(loss.item(),result.loss(),2e-6); assertEquals(.75,result.score(),1e-7); assertEquals(4,result.weight());
            }
        }
    }

    @Test void checkpointResumesMidEpochWithBitIdenticalWeightsMomentsDropoutAndMetrics() throws Exception {
        Path interrupted = temporary.resolve("interrupted.nnj"), finalA = temporary.resolve("a.nnj"), finalB = temporary.resolve("b.nnj");
        try (var context = ExecutionContext.cpu(); var data = dataset(context); var a = network().build(context); var b = network().build(context);
             var first = loader(context,data); var second = loader(context,data); var validation = DataLoader.builder(context,data).batchSize(4).build();
             var original = Trainer.builder().model(a).optimizer(adam()).clipGlobalNorm(.4).checkpointEvery(2,interrupted).build();
             var resumed = trainer(b)) {
            a.training(false); b.training(false);
            a.children().get("hidden").training(true); b.children().get("hidden").training(true);
            var partial = original.fit(first,validation,3,2);
            assertEquals(Trainer.StopReason.BATCH_LIMIT,partial.reason()); assertEquals(4,first.state().cursor());
            resumed.restore(interrupted,second); assertEquals(original.progress(),resumed.progress());
            var completeA = original.fit(first,validation,3); var completeB = resumed.fit(second,validation,3);
            sameWeights(a,b); assertEquals(completeA,completeB); assertEquals(first.state(),second.state());
            original.save(finalA,first); resumed.save(finalB,second);
            assertArrayEquals(Files.readAllBytes(finalA),Files.readAllBytes(finalB));
        }
    }

    @Test void validationAndEarlyStoppingStateSurviveAnEpochBoundaryCheckpoint() throws Exception {
        Path path = temporary.resolve("early.nnj");
        try (var context = ExecutionContext.cpu(); var data = dataset(context); var a = network().build(context); var b = network().build(context);
             var first = loader(context,data); var second = loader(context,data); var validation = loader(context,data);
             var original = Trainer.builder().model(a).optimizer(Optimizers.sgd().learningRate(0).define()).earlyStopping(2,100).checkpointEvery(1,path).build();
             var resumed = Trainer.builder().model(b).optimizer(Optimizers.sgd().learningRate(0).define()).earlyStopping(2,100).build()) {
            var result = original.fit(first,validation,20);
            assertEquals(Trainer.StopReason.EARLY_STOPPING,result.reason()); assertEquals(3,result.progress().completedEpochs());
            resumed.restore(path,second); sameWeights(a,b);
            var stopped = resumed.fit(second,validation,20);
            assertEquals(result.progress(),stopped.progress()); assertEquals(Trainer.StopReason.EARLY_STOPPING,stopped.reason());
        }
    }

    @Test void callbacksObserveCommittedCountersAndCanStopWithoutAnExtraUpdate() throws Exception {
        var events = new ArrayList<Trainer.Event>();
        try (var context = ExecutionContext.cpu(); var data = dataset(context); var model = network().build(context); var loader = loader(context,data);
             var trainer = Trainer.builder().model(model).callback(event -> {
                 events.add(event); return event.progress().batches()==2 ? Trainer.Decision.STOP : Trainer.Decision.CONTINUE;
             }).build()) {
            var result = trainer.fit(loader,10);
            assertEquals(Trainer.StopReason.REQUESTED,result.reason()); assertEquals(2,result.progress().steps());
            assertEquals(2,events.size()); assertEquals(4,loader.state().cursor());
            trainer.fit(loader,1); assertEquals(1,trainer.progress().completedEpochs());
        }
    }

    @Test void crossThreadStopIsCooperativeAndCallbacksCannotReenterOrCloseTheTrainer() throws Exception {
        try (var context = ExecutionContext.cpu(); var data = dataset(context); var model = network().build(context); var loader = loader(context,data)) {
            Trainer[] owner = new Trainer[1];
            try (var trainer = Trainer.builder().model(model).callback(event -> {
                assertThrows(IllegalStateException.class, () -> owner[0].evaluate(loader));
                assertThrows(IllegalStateException.class, () -> owner[0].close());
                return Trainer.Decision.STOP;
            }).build()) {
                owner[0]=trainer;
                try (var executor = java.util.concurrent.Executors.newSingleThreadExecutor()) { executor.submit(trainer::requestStop).get(); }
                assertEquals(0,trainer.fit(loader,1).progress().steps());
                assertEquals(1,trainer.fit(loader,1).progress().steps());
            }
        }
    }

    @Test void failedObjectiveRollsBackLoaderAndDropoutBeforeRetry() throws Exception {
        var fail = new AtomicBoolean(true);
        TrainingObjective objective = new TrainingObjective() {
            public String id() { return "retry-fixture"; }
            public long weight(Tensor target) { return target.shape().elements(); }
            public Tensor loss(Tensor prediction, Tensor target) {
                if (fail.getAndSet(false)) { throw new IllegalArgumentException("Injected objective failure"); }
                return Losses.crossEntropy(prediction,target);
            }
        };
        try (var context = ExecutionContext.cpu(); var data = dataset(context); var model = network().build(context); var loader = loader(context,data);
             var trainer = Trainer.builder().model(model).objective(objective).build()) {
            var state = StateDict.capture(model); var cursor = loader.state(); int allocations = context.liveAllocations();
            assertThrows(IllegalArgumentException.class, () -> trainer.fit(loader,1));
            assertEquals(state.trainingState(),model.trainingState()); assertEquals(cursor,loader.state());
            assertEquals(allocations,context.liveAllocations()); assertEquals(0,trainer.progress().steps());
            trainer.fit(loader,1); assertEquals(3,trainer.progress().steps());
        }
    }

    @Test void allIgnoredBatchesAdvanceTheEpochWithoutUpdatingWeightsOrSchedule() throws Exception {
        try (var context = ExecutionContext.cpu(); var model = new Linear(context,1,2,42);
             var input = context.tensor(new float[]{1,2,3},Shape.of(3,1)); var target = context.tensor(new int[]{-100,-100,-100},Shape.of(3));
             var data = new TensorDataset(Map.of("input",input,"target",target)); var loader = DataLoader.builder(context,data).batchSize(2).build();
             var trainer = Trainer.builder().model(model).build()) {
            float[] before;
            try (var weight = model.namedParameters().get("weight").value()) { before = weight.toFloatArray(); }
            var result = trainer.fit(loader,2);
            assertEquals(new Trainer.Progress(2,0,4),result.progress()); assertEquals(0,result.training().weight());
            assertTrue(Double.isNaN(trainer.evaluate(loader).score()));
            try (var weight = model.namedParameters().get("weight").value()) { assertArrayEquals(before,weight.toFloatArray()); }
        }
    }

    @Test void evaluationFailureRestoresItsCursorAndModeWithoutAdvancingDropout() {
        TrainingObjective failure = new TrainingObjective() {
            public String id() { return "evaluation-failure"; }
            public long weight(Tensor target) { return target.shape().elements(); }
            public Tensor loss(Tensor prediction, Tensor target) { throw new IllegalArgumentException("Injected evaluation failure"); }
        };
        try (var context = ExecutionContext.cpu(); var data = dataset(context); var model = network().build(context); var loader = loader(context,data);
             var trainer = Trainer.builder().model(model).objective(failure).build()) {
            try (var batch = loader.next()) { }
            var state = loader.state(); var modes = model.trainingState(); int allocations = context.liveAllocations();
            assertThrows(IllegalArgumentException.class, () -> trainer.evaluate(loader));
            assertEquals(state,loader.state()); assertEquals(modes,model.trainingState()); assertEquals(allocations,context.liveAllocations());
        }
    }

    @Test void callbackFailureKeepsTheBatchCommittedAndRetryCompletesOnlyTheRemainingBatches() throws Exception {
        var fail = new AtomicBoolean(true);
        try (var context = ExecutionContext.cpu(); var data = dataset(context); var model = network().build(context); var loader = loader(context,data);
             var trainer = Trainer.builder().model(model).callback(event -> {
                 if (fail.getAndSet(false)) { throw new IllegalArgumentException("Injected callback failure"); }
                 return Trainer.Decision.CONTINUE;
             }).build()) {
            model.training(false);
            assertThrows(IllegalArgumentException.class, () -> trainer.fit(loader,1));
            assertEquals(new Trainer.Progress(0,1,1),trainer.progress()); assertEquals(2,loader.state().cursor()); assertFalse(model.isTraining());
            trainer.fit(loader,1); assertEquals(new Trainer.Progress(1,3,3),trainer.progress());
            try (var batch = loader.next()) { }
            assertThrows(IllegalArgumentException.class, () -> trainer.fit(loader,1));
            assertEquals(3,trainer.progress().steps());
        }
    }

    @Test void failedEpochValidationRequiresRestoreBeforeAnotherTrainingUpdate() throws Exception {
        Path path = temporary.resolve("before-validation.nnj");
        var fail = new AtomicBoolean(true);
        try (var context = ExecutionContext.cpu(); var data = dataset(context); var model = network().build(context);
             var loader = loader(context,data); var validation = loader(context,data)) {
            TrainingObjective objective = new TrainingObjective() {
                public String id() { return "validation-failure"; }
                public long weight(Tensor target) { return target.shape().elements(); }
                public Tensor loss(Tensor prediction, Tensor target) {
                    if (!model.isTraining() && fail.getAndSet(false)) { throw new IllegalArgumentException("Injected epoch validation failure"); }
                    return Losses.crossEntropy(prediction,target);
                }
            };
            try (var trainer = Trainer.builder().model(model).objective(objective).earlyStopping(2,0).build()) {
                trainer.save(path,loader);
                assertThrows(IllegalArgumentException.class, () -> trainer.fit(loader,validation,2));
                assertEquals(3,trainer.progress().steps());
                assertThrows(IllegalStateException.class, () -> trainer.fit(loader,validation,1));
                assertThrows(IllegalStateException.class, () -> trainer.save(path,loader));
                trainer.restore(path,loader);
                trainer.fit(loader,validation,1); assertEquals(3,trainer.progress().steps());
            }
        }
    }

    @Test void callbacksReportStepThenEpochAndValidationAndNonfiniteLossCannotUpdate() throws Exception {
        var events = new ArrayList<Trainer.EventKind>();
        try (var context = ExecutionContext.cpu(); var data = dataset(context); var model = network().build(context); var loader = loader(context,data);
             var validation = loader(context,data); var trainer = Trainer.builder().model(model).callback(event -> {
                 events.add(event.kind()); return Trainer.Decision.CONTINUE;
             }).build()) {
            trainer.fit(loader,validation,1);
            assertEquals(List.of(Trainer.EventKind.STEP,Trainer.EventKind.STEP,Trainer.EventKind.STEP,Trainer.EventKind.EPOCH,Trainer.EventKind.VALIDATION),events);
            TrainingObjective nonfinite = new TrainingObjective() {
                public String id() { return "nonfinite"; }
                public long weight(Tensor target) { return 1; }
                public Tensor loss(Tensor prediction, Tensor target) { return context.tensor(new float[]{Float.NaN},Shape.of()); }
            };
            var before = StateDict.capture(model); var position = loader.state();
            try (var bad = Trainer.builder().model(model).objective(nonfinite).build()) {
                assertThrows(ArithmeticException.class, () -> bad.fit(loader,1));
                assertEquals(0,bad.progress().steps()); assertEquals(position,loader.state());
                try (var reference = network().build(context)) { before.applyTo(reference); sameWeights(model,reference); }
            }
        }
    }

    @Test void incompatibleOrCorruptLoopStateIsRejectedBeforeAnyRestoreMutation() throws Exception {
        Path good = temporary.resolve("good.nnj"), bad = temporary.resolve("bad.nnj");
        try (var context = ExecutionContext.cpu(); var data = dataset(context); var a = network().build(context); var b = network().build(context);
             var first = loader(context,data); var second = loader(context,data); var original = trainer(a); var target = trainer(b);
             var incompatible = Trainer.builder().model(b).optimizer(adam()).clipGlobalNorm(2).build()) {
            original.fit(first,null,2,2); original.save(good,first);
            var state = Checkpoint.read(good); var metadata = new LinkedHashMap<>(state.metadata()); metadata.put("trainer.epochBatches","1");
            Checkpoint.write(bad,new Checkpoint.Loaded(state.model(),state.optimizer(),metadata));
            var before = StateDict.capture(b); var cursor = second.state();
            assertThrows(IllegalArgumentException.class, () -> target.restore(bad,second));
            assertThrows(IllegalArgumentException.class, () -> incompatible.restore(good,second));
            assertEquals(before.trainingState(),b.trainingState()); assertEquals(cursor,second.state()); assertEquals(0,target.progress().steps());
            try (var beforeModel = network().build(context)) { before.applyTo(beforeModel); sameWeights(beforeModel,b); }
            target.restore(good,second); sameWeights(a,b);
        }
    }

    @Test void updateFailureRequiresRestoreAndDoesNotProduceAnUnrecoverableCheckpoint() throws Exception {
        Path path = temporary.resolve("before.nnj"); var fail = new AtomicBoolean(true);
        OptimizerFactory factory = parameters -> {
            var delegate = adam().build(parameters);
            return new Optimizer() {
                public void step() { delegate.step(); if (fail.getAndSet(false)) { throw new IllegalStateException("Injected failure after update"); } }
                public void zeroGrad() { delegate.zeroGrad(); }
                public long steps() { return delegate.steps(); }
                public OptimizerState state() { return delegate.state(); }
                public void validateState(OptimizerState state) { delegate.validateState(state); }
                public void restoreState(OptimizerState state) { delegate.restoreState(state); }
                public void close() { delegate.close(); }
            };
        };
        try (var context = ExecutionContext.cpu(); var data = dataset(context); var model = network().build(context); var loader = loader(context,data);
             var trainer = Trainer.builder().model(model).optimizer(factory).build()) {
            trainer.save(path,loader);
            assertThrows(IllegalStateException.class, () -> trainer.fit(loader,1));
            assertThrows(IllegalStateException.class, () -> trainer.save(path,loader));
            assertThrows(IllegalStateException.class, () -> trainer.fit(loader,1));
            trainer.restore(path,loader); assertEquals(0,loader.state().cursor());
            trainer.fit(loader,1); assertEquals(3,trainer.progress().steps());
        }
    }

    @Test void trainerAlsoTrainsCausalLanguageModelsThroughTheSameModuleBoundary() throws Exception {
        var definition = CausalLanguageModel.builder().vocabularySize(5).modelWidth(8).blockCount(2).contextLength(8)
                .attention(a -> a.queryHeads(2).keyValueHeads(1)).feedForward(f -> f.hiddenWidth(12)).seed(42).define();
        try (var context = ExecutionContext.cpu(); var model = definition.build(context);
             var input = context.tensor(new int[]{0,1,2,1,2,3},Shape.of(2,3)); var target = context.tensor(new int[]{1,2,3,2,3,4},Shape.of(2,3));
             var data = new TensorDataset(Map.of("tokens",input,"labels",target)); var loader = DataLoader.builder(context,data).batchSize(2).build();
             var trainer = Trainer.builder().model(model).fields("tokens","labels").optimizer(Optimizers.adamW().learningRate(.025f).define()).build()) {
            double before = trainer.evaluate(loader).loss();
            trainer.fit(loader,25); var result = trainer.evaluate(loader);
            assertTrue(result.loss()<before*.4); assertEquals(6,result.weight());
        }
    }

    @Test void closureBorrowsModelAndDataAndGuardsTapeAndForeignContextUse() throws Exception {
        try (var context = ExecutionContext.cpu(); var other = ExecutionContext.cpu(); var data = dataset(context); var foreign = dataset(other);
             var model = network().build(context); var loader = loader(context,data); var wrong = loader(other,foreign)) {
            int before = context.liveAllocations();
            var trainer = trainer(model);
            assertThrows(IllegalArgumentException.class, () -> trainer.fit(wrong,1));
            try (var tape = context.gradientTape()) { assertThrows(IllegalStateException.class, () -> trainer.evaluate(loader)); }
            trainer.close(); assertEquals(before,context.liveAllocations());
            assertThrows(IllegalStateException.class,trainer::progress); assertEquals(5,data.size());
            try (var input = context.tensor(new float[]{0,1},Shape.of(1,2)); var output = model.forward(input)) { assertEquals(Shape.of(1,2),output.shape()); }
        }
    }
}

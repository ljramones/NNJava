package org.dynamisengine.nn;

import org.dynamisengine.nn.data.DataLoader;
import org.dynamisengine.nn.data.TensorDataset;
import org.dynamisengine.nn.io.StateDict;
import org.dynamisengine.nn.nn.*;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;
import org.dynamisengine.nn.training.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

class CpuTrainingAndStateTest {
    @TempDir
    Path temporary;

    private static Sequential dropoutModel(ExecutionContext c, long seed) {
        return new Sequential(c, new Linear(c, 2, 4, seed), new Dropout(c, .25f, seed + 1), Activation.tanh(c), new Linear(c, 4, 2, seed + 2));
    }

    static float step(ExecutionContext c, UnaryModule model, Optimizer optimizer, Tensor x, Tensor y) {
        optimizer.zeroGrad();
        float value;
        try (var tape = c.gradientTape(); var prediction = model.forward(x); var loss = Losses.crossEntropy(prediction, y)) {
            value = loss.item();
            tape.backward(loss);
        }
        optimizer.step();
        return value;
    }

    private static float batchStep(ExecutionContext c, UnaryModule model, Optimizer optimizer, DataLoader loader) {
        if (!loader.hasNext()) {
            loader.nextEpoch();
        }
        try (var batch = loader.next(); var x = batch.get("x"); var y = batch.get("y")) {
            return step(c, model, optimizer, x, y);
        }
    }

    @Test
    void nonlinearSequentialClassifierLearnsXorUsingAdamW() {
        try (var c = ExecutionContext.cpu(); var model = new Sequential(c, new Linear(c, 2, 8, 11), Activation.tanh(c), new Linear(c, 8, 2, 12));
             var optimizer = new AdamW(model.namedParameters(), .03f);
             var x = c.tensor(new float[]{0, 0, 0, 1, 1, 0, 1, 1}, Shape.of(4, 2)); var labels = c.tensor(new int[]{0, 1, 1, 0}, Shape.of(4))) {
            for (int i = 0; i < 250; i++) {
                step(c, model, optimizer, x, labels);
            }
            model.training(false);
            assertTrue(model.children().values().stream().noneMatch(org.dynamisengine.nn.nn.Module::isTraining));
            try (var prediction = model.forward(x); var loss = Losses.crossEntropy(prediction, labels)) {
                assertTrue(loss.item() < .02f);
                assertEquals(1, Metrics.accuracy(prediction, labels, -100));
            }
            optimizer.zeroGrad();
            assertEquals(4, model.namedParameters().size());
        }
    }

    @Test
    void optimizerUpdatesMatchIndependentScalarReferencesAndPreflightNonfiniteGradients() {
        try (var c = ExecutionContext.cpu(); var leaf = c.tensor(new float[]{2}, Shape.of(), true); var p = new Parameter("p", leaf);
             var optimizer = new AdamW(List.of(new ParameterGroup(Map.of("p", p), .01f, .1f)), .9f, .999f, 1e-8f, LearningRateSchedule.constant())) {
            try (var tape = c.gradientTape(); var loss = Ops.square(leaf)) {
                tape.backward(loss);
            }
            optimizer.step();
            assertEquals(1.988f, leaf.item(), 2e-6f);
            assertEquals(1, optimizer.steps());
        }
        try (var c = ExecutionContext.cpu(); var leaf = c.tensor(new float[]{2}, Shape.of(), true); var p = new Parameter("p", leaf);
             var optimizer = new Sgd(List.of(p), .1f, .9f, 0, false)) {
            for (int i = 0; i < 2; i++) {
                optimizer.zeroGrad();
                try (var tape = c.gradientTape(); var loss = Ops.square(leaf)) {
                    tape.backward(loss);
                }
                optimizer.step();
            }
            assertEquals(.92f, leaf.item(), 2e-6f);
        }
        try (var c = ExecutionContext.cpu(); var leaf = c.tensor(new float[]{2}, Shape.of(), true); var zero = c.full(Shape.of(), 0);
             var p = new Parameter("p", leaf); var optimizer = new AdamW(List.of(p), .01f)) {
            try (var tape = c.gradientTape(); var loss = Ops.divide(leaf, zero)) {
                tape.backward(loss);
            }
            assertThrows(ArithmeticException.class, optimizer::step);
            assertEquals(2, leaf.item());
            assertEquals(0, optimizer.steps());
        }
    }

    @Test
    void gradientClippingDeduplicatesSharedParametersAndMakesWritableGradients() {
        try (var c = ExecutionContext.cpu(); var leaf = c.tensor(new float[]{1, 1}, Shape.of(2), true); var weights = c.tensor(new float[]{3, 4}, Shape.of(2));
             var p = new Parameter("p", leaf); var alias = new Parameter("alias", leaf); var tape = c.gradientTape(); var product = Ops.multiply(leaf, weights); var loss = Ops.sum(product)) {
            tape.backward(loss);
            assertEquals(5, Gradients.clipGlobalNorm(List.of(p, alias), 1), 1e-6);
            try (var gradient = leaf.gradient()) {
                assertArrayEquals(new float[]{.6f, .8f}, gradient.toFloatArray(), 1e-6f);
            }
        }
    }

    @Test
    void checkpointResumesAdamMomentsDropoutAndDataOrderBitForBit() throws Exception {
        Path file = temporary.resolve("training.nnjava");
        try (var c = ExecutionContext.cpu(); var x = c.tensor(new float[]{0, 0, 0, 1, 1, 0, 1, 1}, Shape.of(4, 2)); var labels = c.tensor(new int[]{0, 1, 1, 0}, Shape.of(4));
             var dataset = new TensorDataset(Map.of("x", x, "y", labels));
             var loader = new DataLoader(c, dataset, 2, true, false, 77); var resumedLoader = new DataLoader(c, dataset, 2, true, false, 77);
             var model = dropoutModel(c, 10); var restored = dropoutModel(c, 999);
             var optimizer = new AdamW(model.namedParameters(), .01f); var resumedOptimizer = new AdamW(restored.namedParameters(), .01f)) {
            for (int i = 0; i < 7; i++) {
                batchStep(c, model, optimizer, loader);
            }
            Checkpoint.save(file, model, optimizer, loader.state().metadata());
            var loaded = Checkpoint.read(file);
            loaded.restore(restored, resumedOptimizer);
            resumedLoader.restore(DataLoader.State.fromMetadata(loaded.metadata()));
            assertEquals(model.trainingState(), restored.trainingState());
            for (int i = 0; i < 8; i++) {
                assertEquals(Float.floatToIntBits(batchStep(c, model, optimizer, loader)), Float.floatToIntBits(batchStep(c, restored, resumedOptimizer, resumedLoader)));
            }
            for (String name : model.namedParameters().keySet()) {
                try (var a = model.namedParameters().get(name).value(); var b = restored.namedParameters().get(name).value()) {
                    assertArrayEquals(a.toFloatArray(), b.toFloatArray());
                }
            }
            assertEquals(optimizer.state().counters(), resumedOptimizer.state().counters());
        }
    }

    @Test
    void checkpointPreservesAliasesAndRejectsCorruptionOrMismatchedArchitecture() throws Exception {
        Path file = temporary.resolve("weights.nnjava");
        try (var c = ExecutionContext.cpu(); var model = new Tied(c, true, 4); var restored = new Tied(c, true, 55); var untied = new Tied(c, false, 66)) {
            var state = StateDict.capture(model);
            assertEquals(1, state.tensors().size());
            assertEquals(1, state.aliases().size());
            assertThrows(IllegalArgumentException.class, () -> state.applyTo(untied));
            Checkpoint.saveWeights(file, model);
            Checkpoint.read(file).restoreModel(restored);
            try (var a = model.embedding.weight().value(); var b = restored.embedding.weight().value()) {
                assertArrayEquals(a.toFloatArray(), b.toFloatArray());
            }
            byte[] bytes = Files.readAllBytes(file);
            bytes[bytes.length - 9] ^= 1;
            Files.write(file, bytes);
            assertThrows(java.io.IOException.class, () -> Checkpoint.read(file));
            Files.write(file, Arrays.copyOf(bytes, 12));
            assertThrows(java.io.IOException.class, () -> Checkpoint.read(file));
        }
    }

    @Test
    void dataLoaderRestoresMidEpochAndSupportsDropLast() {
        try (var c = ExecutionContext.cpu(); var values = c.tensor(new int[]{0, 1, 2, 3, 4}, Shape.of(5)); var dataset = new TensorDataset(Map.of("ids", values));
             var loader = new DataLoader(c, dataset, 2, true, false, 12); var copy = new DataLoader(c, dataset, 2, true, false, 12);
             var drop = new DataLoader(c, dataset, 2, false, true, 0)) {
            try (var batch = loader.next()) {
                assertEquals(Set.of("ids"), batch.names());
            }
            copy.restore(loader.state());
            while (loader.hasNext()) {
                try (var a = loader.next(); var b = copy.next(); var av = a.get("ids"); var bv = b.get("ids")) {
                    assertArrayEquals(av.toIntArray(), bv.toIntArray());
                }
            }
            for (int i = 0; i < 2; i++) {
                try (var ignored = drop.next()) {
                }
            }
            assertFalse(drop.hasNext());
            assertThrows(NoSuchElementException.class, drop::next);
            var state = loader.state();
            loader.nextEpoch();
            assertNotEquals(state.epoch(), loader.state().epoch());
        }
    }

    @Test
    void schedulesAndSharedGroupValidationHaveDefinedBehavior() {
        var schedule = LearningRateSchedule.warmupCosine(2, 6, .1f);
        assertEquals(.5f, schedule.rate(1, 1));
        assertEquals(1, schedule.rate(2, 1));
        assertEquals(.1f, schedule.rate(6, 1), 1e-7f);
        try (var c = ExecutionContext.cpu(); var value = c.tensor(new float[]{1}, Shape.of(), true); var p = new Parameter("p", value)) {
            assertThrows(IllegalArgumentException.class, () -> new AdamW(List.of(new ParameterGroup(Map.of("p", p), .1f, 0), new ParameterGroup(Map.of("alias", p), .2f, 0)), .9f, .999f, 1e-8f, LearningRateSchedule.constant()));
        }
    }

    @Test
    void checkpointValidatesBehavioralConfigurationBeforeChangingWeights() {
        try (var c = ExecutionContext.cpu(); var source = dropoutModel(c, 1);
             var different = new Sequential(c, new Linear(c, 2, 4, 99), new Dropout(c, .5f, 100), Activation.tanh(c), new Linear(c, 4, 2, 101))) {
            var state = StateDict.capture(source);
            try (var weight = different.namedParameters().get("0.weight").value()) {
                float[] before = weight.toFloatArray();
                assertThrows(IllegalArgumentException.class, () -> state.applyTo(different));
                assertArrayEquals(before, weight.toFloatArray());
            }
        }
    }

    @Test
    void clippingHandlesNormsLargerThanFloatMaximum() {
        try (var c = ExecutionContext.cpu(); var leaf = c.tensor(new float[]{0, 0}, Shape.of(2), true);
             var coefficients = c.tensor(new float[]{3e38f, 3e38f}, Shape.of(2)); var parameter = new Parameter("p", leaf);
             var tape = c.gradientTape(); var product = Ops.multiply(leaf, coefficients); var loss = Ops.sum(product)) {
            tape.backward(loss);
            double norm = Gradients.clipGlobalNorm(List.of(parameter), 1);
            assertTrue(Double.isFinite(norm) && norm > Float.MAX_VALUE);
            try (var gradient = leaf.gradient()) {
                assertArrayEquals(new float[]{(float) Math.sqrt(.5), (float) Math.sqrt(.5)}, gradient.toFloatArray(), 1e-5f);
            }
        }
    }

    private static final class Tied extends AbstractModule {
        final Embedding embedding;

        Tied(ExecutionContext c, boolean tied, long seed) {
            super(c);
            embedding = register("embedding", new Embedding(c, 5, 4, seed));
            register("output", tied ? new Linear(c, embedding.weight(), false) : new Linear(c, 4, 5, seed + 1, false));
        }
    }
}

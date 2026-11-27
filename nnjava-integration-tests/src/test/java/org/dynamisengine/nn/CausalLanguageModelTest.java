package org.dynamisengine.nn;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import org.dynamisengine.nn.backend.Backend;
import org.dynamisengine.nn.backend.vectrix.VectrixBackend;
import org.dynamisengine.nn.io.ModelDefinitions;
import org.dynamisengine.nn.io.StateArchive;
import org.dynamisengine.nn.io.StateDict;
import org.dynamisengine.nn.model.ComponentSpec;
import org.dynamisengine.nn.nn.*;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;
import org.dynamisengine.nn.training.AdamW;
import org.dynamisengine.nn.training.Checkpoint;
import org.dynamisengine.nn.training.Losses;
import org.dynamisengine.nn.training.Metrics;
import org.dynamisengine.nn.transformer.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class CausalLanguageModelTest {
    @TempDir Path temporary;

    private static CausalLanguageModel.Builder tiny() {
        return CausalLanguageModel.builder().name("tiny").vocabularySize(5).modelWidth(4).blockCount(2).contextLength(8)
                .attention(a -> a.queryHeads(2).keyValueHeads(1)).feedForward(f -> f.hiddenWidth(8)).seed(42);
    }

    @Test void definitionsRoundTripWithoutAllocatingAndSnapshotNestedBuilderSettings() throws Exception {
        var builder = tiny(); var definition = builder.define();
        builder.attention(a -> a.queryHeads(1)).feedForward(f -> f.hiddenWidth(12).gelu());
        assertEquals(2, definition.block().attention().heads());
        assertEquals(FeedForwardKind.SWIGLU, definition.block().feedForward());
        Path file = temporary.resolve("decoder.nnjd");
        ModelDefinitions.write(file, definition.specification());
        var restored = CausalLanguageModel.Definition.fromSpec(ModelDefinitions.read(file));
        assertEquals(definition, restored);
        try (var context = ExecutionContext.cpu()) { restored.validate(context); assertEquals(0, context.liveAllocations()); }
    }

    @Test void rejectsInconsistentHeadsPositionsAndOtherSettingsBeforeAllocation() {
        assertThrows(IllegalArgumentException.class, () -> tiny().modelWidth(5).define());
        assertThrows(IllegalArgumentException.class, () -> tiny().attention(a -> a.queryHeads(3)).define());
        assertThrows(IllegalArgumentException.class, () -> tiny().attention(a -> a.keyValueHeads(3)).define());
        assertThrows(IllegalArgumentException.class, () -> tiny().attention(a -> a.keyValueHeads(0)).define());
        assertThrows(IllegalArgumentException.class, () -> tiny().modelWidth(6).define());
        assertThrows(IllegalArgumentException.class, () -> tiny().blockCount(0).define());
        assertThrows(IllegalArgumentException.class, () -> tiny().contextLength(0).define());
        assertThrows(IllegalArgumentException.class, () -> tiny().normalizationEpsilon(Float.NaN).define());
        assertThrows(IllegalArgumentException.class, () -> tiny().embeddingDropout(1).define());
        assertThrows(IllegalArgumentException.class, () -> tiny().feedForward(f -> f.hiddenWidth(0)).define());
        var definition = tiny().define(); var spec = definition.specification();
        assertThrows(IllegalArgumentException.class, () -> CausalLanguageModel.Definition.fromSpec(
                new ComponentSpec(spec.type(), 99, spec.name(), spec.attributes(), spec.children())));
    }

    @Test void blocksHaveDistinctWeightsAndReadoutTiesArePreserved() throws Exception {
        var definition = tiny().define();
        try (var context = ExecutionContext.cpu(); var model = definition.build(context); var restored = definition.build(context);
                var first = model.namedParameters().get("block0.attention.query.weight").value();
                var second = model.namedParameters().get("block1.attention.query.weight").value()) {
            assertFalse(Arrays.equals(first.toFloatArray(), second.toFloatArray()));
            assertEquals("embedding.weight", StateDict.capture(model).aliases().get("output.weight"));
            Path file = temporary.resolve("model.nnj"); StateArchive.saveWeights(file, model); StateArchive.read(file).restoreModel(restored);
            assertEquals("embedding.weight", StateDict.capture(restored).aliases().get("output.weight"));
            try (var tokens = context.tensor(new int[] {0, 1, 2}, Shape.of(1, 3)); var a = model.forward(tokens); var b = restored.forward(tokens)) {
                assertEquals(Shape.of(1, 3, 5), a.shape()); assertArrayEquals(a.toFloatArray(), b.toFloatArray());
            }
        }
    }

    @Test void configurableNormalizationsAndFeedForwardFamiliesRunWithUntiedReadout() {
        for (NormalizationKind normalization : NormalizationKind.values()) {
            for (FeedForwardKind kind : FeedForwardKind.values()) {
                var definition = tiny().normalization(normalization).tieInputOutputEmbeddings(false).feedForward(f -> {
                    switch (kind) { case SWIGLU -> f.swiGlu(); case GELU -> f.gelu(); case SILU -> f.silu(); case RELU -> f.relu(); }
                }).define();
                try (var context = ExecutionContext.cpu(); var model = definition.build(context);
                        var tokens = context.tensor(new int[] {0, 1}, Shape.of(1, 2)); var logits = model.forward(tokens)) {
                    assertEquals(Shape.of(1, 2, 5), logits.shape());
                    assertTrue(StateDict.capture(model).aliases().isEmpty());
                    for (float value : logits.toFloatArray()) { assertTrue(Float.isFinite(value)); }
                }
            }
        }
    }

    @Test void decoderMatchesIndependentCompositionOfTheExistingPublicPrimitives() {
        var definition = tiny().blockCount(1).finalNormalization(false).define();
        try (var context = ExecutionContext.cpu(); var model = definition.build(context); var manual = new ManualDecoder(context, definition);
                var tokens = context.tensor(new int[] {0, 1, 2, 1, 2, 3}, Shape.of(2, 3));
                var expected = manual.forward(tokens); var actual = model.forward(tokens)) {
            assertArrayEquals(expected.toFloatArray(), actual.toFloatArray());
        }
    }

    @Test void cachedPrefillAndChunkedDecodeMatchFullForwardForMultipleLayersAndBatches() {
        var definition = tiny().contextLength(4).define();
        try (var context = ExecutionContext.cpu(); var model = definition.build(context);
                var tokens = context.tensor(new int[] {0, 1, 2, 3, 1, 2, 3, 4}, Shape.of(2, 4))) {
            model.training(false);
            var before = model.trainingState();
            try (var session = model.newSession(2); var full = model.forward(tokens);
                    var prefix = Ops.slice(tokens, 1, 0, 2); var suffix = Ops.slice(tokens, 1, 2, 4);
                    var prefill = session.prefill(prefix); var decode = session.decode(suffix);
                    var expectedPrefix = Ops.slice(full, 1, 0, 2); var expectedSuffix = Ops.slice(full, 1, 2, 4)) {
                assertArrayEquals(expectedPrefix.toFloatArray(), prefill.toFloatArray(), 2e-5f);
                assertArrayEquals(expectedSuffix.toFloatArray(), decode.toFloatArray(), 2e-5f);
                assertEquals(4, session.length());
                assertThrows(IllegalArgumentException.class, () -> session.decode(suffix));
                assertEquals(4, session.length());
                session.reset(); assertEquals(0, session.length());
                try (var repeated = session.prefill(prefix)) { assertArrayEquals(prefill.toFloatArray(), repeated.toFloatArray()); }
            }
            assertEquals(before, model.trainingState());
        }
    }

    @Test void mhaMqaGqaAndDisabledRotaryUseTheSameSessionContract() {
        for (int kvHeads : new int[] {1, 2, 4}) {
            var definition = tiny().modelWidth(8).attention(a -> a.queryHeads(4).keyValueHeads(kvHeads)).define();
            checkSessionParity(definition);
        }
        checkSessionParity(tiny().modelWidth(6).attention(a -> a.rotaryPositions(false)).define());
    }

    private static void checkSessionParity(CausalLanguageModel.Definition definition) {
        try (var context = ExecutionContext.cpu(); var model = definition.build(context);
                var tokens = context.tensor(new int[] {0, 1, 2}, Shape.of(1, 3))) {
            model.training(false);
            try (var session = model.newSession(1); var full = model.forward(tokens);
                    var prefix = Ops.slice(tokens, 1, 0, 2); var suffix = Ops.slice(tokens, 1, 2, 3);
                    var ignored = session.prefill(prefix); var actual = session.decode(suffix); var expected = Ops.slice(full, 1, 2, 3)) {
                assertArrayEquals(expected.toFloatArray(), actual.toFloatArray(), 2e-5f);
            }
        }
    }

    @Test void explicitMasksAreValidatedBeforeAnyCacheMutation() {
        try (var context = ExecutionContext.cpu(); var model = tiny().build(context);
                var tokens = context.tensor(new int[] {0, 1, 2}, Shape.of(1, 3));
                var prefix = Ops.slice(tokens, 1, 0, 2); var token = Ops.slice(tokens, 1, 2, 3);
                var bad = context.tensor(new boolean[4], Shape.of(4));
                var allowed = context.tensor(new boolean[] {true, false, true}, Shape.of(1, 1, 1, 3))) {
            model.training(false);
            try (var session = model.newSession(1); var ignored = session.prefill(prefix)) {
                assertThrows(IllegalArgumentException.class, () -> session.decode(token, bad)); assertEquals(2, session.length());
                try (var logits = session.decode(token, allowed)) {
                    for (float value : logits.toFloatArray()) { assertTrue(Float.isFinite(value)); }
                }
                assertEquals(3, session.length());
            }
        }
    }

    @Test void sessionsOwnIndependentCachesAndLeaveBorrowedModelWeightsAlive() {
        try (var context = ExecutionContext.cpu(); var model = tiny().build(context);
                var tokens = context.tensor(new int[] {0, 1}, Shape.of(1, 2))) {
            model.training(false); int before = context.liveAllocations();
            try (var first = model.newSession(1); var second = model.newSession(1); var ignored = first.prefill(tokens)) {
                assertEquals(before + 8, context.liveAllocations() - 1);
                assertEquals(2, first.length()); assertEquals(0, second.length());
                first.reset(); assertEquals(0, first.length());
            }
            assertEquals(before, context.liveAllocations());
            try (var output = model.forward(tokens)) { assertEquals(Shape.of(1, 2, 5), output.shape()); }
        }
    }

    @Test void sessionModeAndTapeGuardsDoNotChangeSharedModelMode() {
        try (var context = ExecutionContext.cpu(); var model = tiny().build(context);
                var tokens = context.tensor(new int[] {0}, Shape.of(1, 1))) {
            int before = context.liveAllocations();
            assertThrows(IllegalStateException.class, () -> model.newSession(1)); assertTrue(model.isTraining());
            assertEquals(before, context.liveAllocations()); model.training(false);
            try (var session = model.newSession(1)) {
                try (var tape = context.gradientTape()) {
                    assertThrows(IllegalStateException.class, () -> session.prefill(tokens)); assertEquals(0, session.length());
                }
                try (var ignored = session.prefill(tokens)) { assertEquals(1, session.length()); }
                model.training(true); assertThrows(IllegalStateException.class, () -> session.decode(tokens));
                model.training(false); try (var ignored = session.decode(tokens)) { assertEquals(2, session.length()); }
            }
            assertFalse(model.isTraining());
        }
    }

    @Test void staleCachesRequireResetAfterWeightsChange() {
        try (var context = ExecutionContext.cpu(); var model = tiny().build(context);
                var tokens = context.tensor(new int[] {0}, Shape.of(1, 1))) {
            model.training(false);
            try (var session = model.newSession(1); var ignored = session.prefill(tokens);
                    var weight = model.namedParameters().get("embedding.weight").value()) {
                context.fill(weight, .125f);
                var error = assertThrows(IllegalStateException.class, () -> session.decode(tokens));
                assertTrue(error.getMessage().contains("reset")); assertEquals(1, session.length());
                session.reset();
                try (var actual = session.prefill(tokens); var expected = model.forward(tokens)) {
                    assertArrayEquals(expected.toFloatArray(), actual.toFloatArray());
                }
            }
        }
    }

    @Test void partialConstructionCleansUpEveryAllocationFailurePoint() {
        var definition = tiny().normalization(NormalizationKind.LAYER).define();
        for (int failAt = 1; failAt <= 25; failAt++) {
            try (var context = ExecutionContext.create(failingBackend(failAt))) {
                assertThrows(IllegalStateException.class, () -> definition.build(context));
                assertEquals(0, context.liveAllocations(), "allocation failure point " + failAt);
            }
        }
    }

    @Test void failedCacheConstructionPreservesModelWeightsAndReleasesTemporaryStorage() {
        var control = controlledBackend(Integer.MAX_VALUE);
        try (var context = ExecutionContext.create(control.backend); var model = tiny().build(context)) {
            model.training(false); int before = context.liveAllocations();
            control.failAt.set(control.count.get() + 2);
            assertThrows(IllegalStateException.class, () -> model.newSession(1));
            assertEquals(before, context.liveAllocations());
            try (var session = model.newSession(1)) { assertEquals(0, session.length()); }
            assertEquals(before, context.liveAllocations());
        }
    }

    @Test void computeFailureInvalidatesTheSessionWhileAHealthyModelCanOpenAnother() {
        var control = controlledBackend(Integer.MAX_VALUE);
        try (var context = ExecutionContext.create(control.backend); var model = tiny().build(context);
                var tokens = context.tensor(new int[] {0}, Shape.of(1, 1))) {
            model.training(false); int before = context.liveAllocations();
            try (var session = model.newSession(1)) {
                control.failAt.set(control.count.get() + 1);
                assertThrows(IllegalStateException.class, () -> session.prefill(tokens));
                assertThrows(IllegalStateException.class, session::reset);
                assertThrows(IllegalStateException.class, () -> session.decode(tokens));
            }
            assertEquals(before, context.liveAllocations());
            try (var fresh = model.newSession(1); var output = fresh.prefill(tokens)) { assertEquals(Shape.of(1, 1, 5), output.shape()); }
        }
    }

    @Test void closingBorrowedModelInvalidatesSessionButSessionCloseStillReleasesItsCaches() {
        try (var context = ExecutionContext.cpu()) {
            var model = tiny().build(context); model.training(false);
            var session = model.newSession(1); model.close();
            assertThrows(IllegalStateException.class, session::length);
            session.close(); session.close(); assertEquals(0, context.liveAllocations());
        }
    }

    @Test void tokenTrainingReducesLossAndFullCheckpointResumeIsBitReproducible() throws Exception {
        var definition = tiny().embeddingDropout(.1f).residualDropout(.1f).attention(a -> a.dropout(.1f)).define();
        try (var context = ExecutionContext.cpu(); var model = definition.build(context); var resumed = definition.build(context);
                var optimizer = new AdamW(model.namedParameters(), .025f); var resumedOptimizer = new AdamW(resumed.namedParameters(), .025f);
                var tokens = context.tensor(new int[] {0, 1, 2, 1, 2, 3}, Shape.of(2, 3));
                var labels = context.tensor(new int[] {1, 2, 3, 2, 3, 4}, Shape.of(2, 3))) {
            float first = trainStep(model, optimizer, tokens, labels);
            for (int i = 0; i < 30; i++) { trainStep(model, optimizer, tokens, labels); }
            Path file = temporary.resolve("training.nnj"); Checkpoint.save(file, model, optimizer, Map.of("step", "31"));
            Checkpoint.read(file).restore(resumed, resumedOptimizer);
            for (int i = 0; i < 4; i++) {
                assertEquals(trainStep(model, optimizer, tokens, labels), trainStep(resumed, resumedOptimizer, tokens, labels));
            }
            for (var entry : model.namedParameters().entrySet()) {
                try (var a = entry.getValue().value(); var b = resumed.namedParameters().get(entry.getKey()).value()) {
                    assertArrayEquals(a.toFloatArray(), b.toFloatArray(), entry.getKey());
                }
            }
            assertEquals(model.trainingState(), resumed.trainingState());
            model.training(false);
            try (var logits = model.forward(tokens); var loss = Losses.crossEntropy(logits, labels)) {
                assertTrue(loss.item() < first * .5f); assertTrue(Metrics.accuracy(logits, labels, -100) >= .8);
            }
        }
    }

    @Test void tiedEmbeddingGradientMatchesFiniteDifferencesThroughTheWholeDecoder() {
        try (var context = ExecutionContext.cpu(); var model = tiny().build(context);
                var tokens = context.tensor(new int[] {0, 1, 0}, Shape.of(1, 3));
                var labels = context.tensor(new int[] {1, 0, 1}, Shape.of(1, 3));
                var weight = model.namedParameters().get("embedding.weight").value()) {
            model.training(false);
            try (var tape = context.gradientTape(); var logits = model.forward(tokens); var loss = Losses.crossEntropy(logits, labels)) { tape.backward(loss); }
            float[] analytic; try (var gradient = weight.gradient()) { analytic = gradient.toFloatArray(); }
            float[] original = weight.toFloatArray();
            for (int index : new int[] {0, 3, 4, 7, 12}) {
                // Resolve the curvature of RMS-normalized, small embeddings while
                // keeping the loss difference large enough for FP32 evaluation.
                float step = .0001f;
                float[] values = original.clone(); values[index] += step; assign(context, weight, values);
                float plus = evaluate(model, tokens, labels);
                values[index] -= 2 * step; assign(context, weight, values);
                float minus = evaluate(model, tokens, labels);
                assertEquals((plus - minus) / (2 * step), analytic[index], .0015f + Math.abs(analytic[index]) * .003f);
            }
            assign(context, weight, original);
        }
    }

    private static float trainStep(CausalLanguageModel model, AdamW optimizer, Tensor tokens, Tensor labels) {
        optimizer.zeroGrad(); float value;
        try (var tape = tokens.context().gradientTape(); var logits = model.forward(tokens); var loss = Losses.crossEntropy(logits, labels)) {
            value = loss.item(); tape.backward(loss);
        }
        optimizer.step(); return value;
    }
    private static float evaluate(CausalLanguageModel model, Tensor tokens, Tensor labels) {
        try (var logits = model.forward(tokens); var loss = Losses.crossEntropy(logits, labels)) { return loss.item(); }
    }
    private static void assign(ExecutionContext context, Tensor target, float[] values) {
        try (var source = context.tensor(values, target.shape())) { context.assign(target, source); }
    }
    private static Backend failingBackend(int failAt) {
        return controlledBackend(failAt).backend;
    }
    private record FaultControl(Backend backend, AtomicInteger count, AtomicInteger failAt) { }
    private static FaultControl controlledBackend(int failurePoint) {
        var cpu = new VectrixBackend(); var count = new AtomicInteger(); var failAt = new AtomicInteger(failurePoint);
        var backend = (Backend) Proxy.newProxyInstance(Backend.class.getClassLoader(), new Class<?>[] {Backend.class}, (proxy, method, args) -> {
            if (method.getName().equals("allocate") && count.incrementAndGet() == failAt.get()) { throw new IllegalStateException("Injected allocation failure"); }
            try { return method.invoke(cpu, args); } catch (InvocationTargetException error) { throw error.getCause(); }
        });
        return new FaultControl(backend, count, failAt);
    }

    private static final class ManualDecoder extends AbstractModule implements UnaryModule {
        private final Embedding embedding;
        private final RmsNorm norm1, norm2;
        private final MultiHeadAttention attention;
        private final GatedFeedForward feedForward;
        private final Linear output;
        ManualDecoder(ExecutionContext context, CausalLanguageModel.Definition definition) {
            super(context); long blockSeed = ModelBuild.seed(definition.seed(), "block0");
            embedding = register("embedding", new Embedding(context, 5, 4, ModelBuild.seed(definition.seed(), "embedding")));
            norm1 = register("norm1", new RmsNorm(context, 4, 1e-5f));
            attention = register("attention", new MultiHeadAttention(context, definition.block().attention(), ModelBuild.seed(blockSeed, "attention")));
            norm2 = register("norm2", new RmsNorm(context, 4, 1e-5f));
            feedForward = register("feedForward", new GatedFeedForward(context, 4, 8, ModelBuild.seed(blockSeed, "feedForward")));
            output = register("output", new Linear(context, embedding.weight(), false));
        }
        @Override public Tensor forward(Tensor tokens) {
            try (var x = embedding.forward(tokens); var n1 = norm1.forward(x); var attended = attention.forward(n1);
                    var residual = Ops.add(x, attended); var n2 = norm2.forward(residual); var feed = feedForward.forward(n2);
                    var hidden = Ops.add(residual, feed)) { return output.forward(hidden); }
        }
    }
}

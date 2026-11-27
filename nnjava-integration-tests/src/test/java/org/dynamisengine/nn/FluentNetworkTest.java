package org.dynamisengine.nn;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import org.dynamisengine.nn.backend.Backend;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.backend.vectrix.VectrixBackend;
import org.dynamisengine.nn.io.ModelDefinitions;
import org.dynamisengine.nn.io.StateArchive;
import org.dynamisengine.nn.model.ComponentSpec;
import org.dynamisengine.nn.nn.*;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;
import org.dynamisengine.nn.training.AdamW;
import org.dynamisengine.nn.training.Losses;
import org.dynamisengine.nn.training.Metrics;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class FluentNetworkTest {
    @TempDir Path temporary;

    @Test void definitionsAreImmutableAndRoundTripWithoutAllocatingWeights() throws Exception {
        try (var context = ExecutionContext.cpu()) {
            var builder = NeuralNetwork.builder().name("classifier").inputFeatures(2).seed(42).dense("hidden", 4, Activation.Kind.TANH);
            var first = builder.define();
            var second = builder.dense("output", 2).define();
            assertEquals(1, first.layers().size()); assertEquals(2, second.layers().size());
            assertThrows(UnsupportedOperationException.class, () -> first.layers().clear());
            Path file = temporary.resolve("model.nnjd");
            ModelDefinitions.write(file, second.specification());
            assertEquals(second, NeuralNetwork.Definition.fromSpec(ModelDefinitions.read(file)));
            assertEquals(0, context.liveAllocations());
            assertEquals(Shape.of(3, 7, 2), second.outputShape(Shape.of(3, 7, 2)));
        }
    }

    @Test void fluentNetworkMatchesDirectCompositionWithTheSameComponentSeeds() {
        long seed = 19;
        var definition = NeuralNetwork.builder().inputFeatures(2).seed(seed)
                .dense("hidden", 4, Activation.Kind.TANH).rmsNorm("norm", 1e-5f).dense("output", 2).define();
        try (var context = ExecutionContext.cpu(); var model = definition.build(context);
                var direct = new Sequential(context, new Linear(context, 2, 4, ModelBuild.seed(seed, "hidden")),
                        Activation.tanh(context), new RmsNorm(context, 4, 1e-5f), new Linear(context, 4, 2, ModelBuild.seed(seed, "output")));
                var input = context.tensor(new float[] {1, -2, .5f, 3}, Shape.of(2, 2));
                var expected = direct.forward(input); var actual = model.forward(input)) {
            assertArrayEquals(expected.toFloatArray(), actual.toFloatArray());
            assertTrue(model.namedParameters().containsKey("hidden.linear.weight"));
        }
    }

    @Test void flattenPreservesBatchPrefixesAndBackwardRestoresInputShape() {
        var definition = NeuralNetwork.builder().inputShape(2, 3).flatten("flat").dense("output", 2).define();
        try (var context = ExecutionContext.cpu(); var model = definition.build(context);
                var input = context.tensor(new float[24], Shape.of(4, 2, 3), true); var tape = context.gradientTape();
                var output = model.forward(input); var loss = Ops.sum(output)) {
            assertEquals(Shape.of(4, 2), output.shape()); tape.backward(loss);
            try (var gradient = input.gradient()) { assertEquals(input.shape(), gradient.shape()); }
        }
        assertThrows(IllegalArgumentException.class, () -> NeuralNetwork.builder().inputFeatures(2).reshape("bad", 3).define());
        assertThrows(IllegalArgumentException.class, () -> NeuralNetwork.builder().inputFeatures(2).dense("same", 3).dense("same", 2).define());
    }

    @Test void stableNamesKeepInitializationIndependentOfParameterlessInsertions() {
        var a = NeuralNetwork.builder().inputFeatures(2).seed(7).dense("hidden", 4).dense("output", 2).define();
        var b = NeuralNetwork.builder().inputFeatures(2).seed(7).dense("hidden", 4).dropout("off", 0).dense("output", 2).define();
        try (var context = ExecutionContext.cpu(); var first = a.build(context); var second = b.build(context);
                var x = first.namedParameters().get("output.linear.weight").value();
                var y = second.namedParameters().get("output.linear.weight").value()) {
            assertArrayEquals(x.toFloatArray(), y.toFloatArray());
        }
    }

    @Test void missingCapabilitiesAreReportedBeforeAnyModelAllocation() {
        AtomicInteger allocations = new AtomicInteger();
        try (var context = ExecutionContext.create(backend(allocations, -1, Operation.SIGMOID))) {
            var definition = NeuralNetwork.builder().name("network").inputFeatures(2).dense("first", 4)
                    .dense("unsupported", 2, Activation.Kind.SIGMOID).define();
            var error = assertThrows(UnsupportedOperationException.class, () -> definition.build(context));
            assertTrue(error.getMessage().contains("network.unsupported"));
            assertEquals(0, allocations.get()); assertEquals(0, context.liveAllocations());
        }
    }

    @Test void failedLayerConstructionRollsBackTheWholePartiallyBuiltNetwork() {
        AtomicInteger allocations = new AtomicInteger();
        try (var context = ExecutionContext.create(backend(allocations, 4, null))) {
            var definition = NeuralNetwork.builder().inputFeatures(2).dense("first", 4).layerNorm("normalization", 1e-5f).define();
            assertThrows(IllegalStateException.class, () -> definition.build(context));
            assertEquals(0, context.liveAllocations());
            try (var usable = context.full(Shape.of(), 2)) { assertEquals(2, usable.item()); }
        }
    }

    @Test void directLayerBuilderValidatesLimitsAndMatchesTheExistingConstructor() {
        var definition = Linear.builder().inputFeatures(2).outputFeatures(3).bias(false).seed(11).define();
        assertEquals(definition, Linear.Definition.fromSpec(definition.specification()));
        try (var context = ExecutionContext.cpu(); var built = definition.build(context); var direct = new Linear(context, 2, 3, 11, false);
                var a = built.namedParameters().get("weight").value(); var b = direct.namedParameters().get("weight").value()) {
            assertArrayEquals(a.toFloatArray(), b.toFloatArray());
            int before = context.liveAllocations();
            assertThrows(IllegalArgumentException.class, () -> Linear.builder().inputFeatures(2).outputFeatures(Integer.MAX_VALUE).build(context));
            assertEquals(before, context.liveAllocations());
        }
    }

    @Test void fluentClassifierLearnsXorAndReloadsThroughTheModelOnlyArchive() throws Exception {
        var definition = NeuralNetwork.builder().name("xor").inputFeatures(2).seed(42)
                .dense("hidden", 8, Activation.Kind.TANH).dense("output", 2).define();
        try (var context = ExecutionContext.cpu(); var model = definition.build(context); var restored = definition.build(context);
                var optimizer = new AdamW(model.namedParameters(), .03f);
                var input = context.tensor(new float[] {0, 0, 0, 1, 1, 0, 1, 1}, Shape.of(4, 2));
                var labels = context.tensor(new int[] {0, 1, 1, 0}, Shape.of(4))) {
            float first = 0, last = 0;
            for (int step = 0; step < 200; step++) {
                optimizer.zeroGrad();
                try (var tape = context.gradientTape(); var logits = model.forward(input); var loss = Losses.crossEntropy(logits, labels)) {
                    last = loss.item(); if (step == 0) { first = last; } tape.backward(loss);
                }
                optimizer.step();
            }
            assertTrue(last < first * .1f);
            model.training(false);
            Path file = temporary.resolve("xor.nnj"); StateArchive.saveWeights(file, model); StateArchive.read(file).restoreModel(restored);
            try (var expected = model.forward(input); var actual = restored.forward(input)) {
                assertEquals(1, Metrics.accuracy(actual, labels, -100));
                assertArrayEquals(expected.toFloatArray(), actual.toFloatArray());
            }
        }
    }

    @Test void customLayerDefinitionsUseExplicitDecoders() {
        var definition = NeuralNetwork.builder().inputFeatures(2).layer("square", new Square()).define();
        assertThrows(IllegalArgumentException.class, () -> NeuralNetwork.Definition.fromSpec(definition.specification()));
        var restored = NeuralNetwork.Definition.fromSpec(definition.specification(), Map.of("example.square", spec -> {
            spec.require("example.square", 1); return new Square();
        }));
        try (var context = ExecutionContext.cpu(); var model = restored.build(context);
                var input = context.tensor(new float[] {2, 3}, Shape.of(2)); var output = model.forward(input)) {
            assertArrayEquals(new float[] {4, 9}, output.toFloatArray());
        }
    }

    @Test void malformedDefinitionsFailBeforeTensorConstructionAndPreserveExistingFiles() throws Exception {
        var definition = NeuralNetwork.builder().inputFeatures(2).dense(2).define();
        var original = definition.specification(); Path file = temporary.resolve("definition.nnjd"); ModelDefinitions.write(file, original);
        ComponentSpec deep = original;
        for (int i = 0; i < 66; i++) { deep = new ComponentSpec("example.node", 1, "node", Map.of(), List.of(deep)); }
        ComponentSpec excessive = deep;
        assertThrows(java.io.IOException.class, () -> ModelDefinitions.write(file, excessive));
        assertEquals(original, ModelDefinitions.read(file));
        byte[] bytes = Files.readAllBytes(file); bytes[12] ^= 1; Files.write(file, bytes);
        assertThrows(java.io.IOException.class, () -> ModelDefinitions.read(file));
        assertThrows(IllegalArgumentException.class, () -> NeuralNetwork.Definition.fromSpec(
                new ComponentSpec(original.type(), 99, original.name(), original.attributes(), original.children())));
    }

    private static Backend backend(AtomicInteger allocations, int failAt, Operation unavailable) {
        var cpu = new VectrixBackend();
        return (Backend) Proxy.newProxyInstance(Backend.class.getClassLoader(), new Class<?>[] {Backend.class}, (proxy, method, args) -> {
            if (method.getName().equals("allocate") && allocations.incrementAndGet() == failAt) { throw new IllegalStateException("Injected allocation failure"); }
            if (method.getName().equals("supports") && args.length == 2 && args[0] == unavailable) { return false; }
            try { return method.invoke(cpu, args); }
            catch (InvocationTargetException error) { throw error.getCause(); }
        });
    }

    private record Square() implements LayerDefinition {
        @Override public Shape outputShape(Shape input) { return input; }
        @Override public void validate(ExecutionContext context, Shape input, String path) { ModelBuild.require(context, path, DType.FP32, Operation.MULTIPLY); }
        @Override public UnaryModule build(ExecutionContext context, Shape input, long seed) { return new SquareLayer(context); }
        @Override public ComponentSpec specification(String name) { return new ComponentSpec("example.square", 1, name, Map.of(), List.of()); }
    }
    private static final class SquareLayer extends AbstractModule implements UnaryModule {
        SquareLayer(ExecutionContext context) { super(context); }
        @Override public Tensor forward(Tensor input) { ensureOpen(); return Ops.square(input); }
    }
}

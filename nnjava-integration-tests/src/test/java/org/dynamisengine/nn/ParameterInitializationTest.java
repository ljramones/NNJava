package org.dynamisengine.nn;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import org.dynamisengine.nn.backend.Backend;
import org.dynamisengine.nn.backend.vectrix.VectrixBackend;
import org.dynamisengine.nn.io.StateDict;
import org.dynamisengine.nn.model.ComponentSpec;
import org.dynamisengine.nn.nn.*;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.transformer.CausalLanguageModel;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ParameterInitializationTest {
    private record Control(Backend backend, AtomicInteger allocations) { }
    private static Control noRandom(int failAt) {
        var cpu = new VectrixBackend(); var count = new AtomicInteger();
        var backend = (Backend) Proxy.newProxyInstance(Backend.class.getClassLoader(), new Class<?>[]{Backend.class}, (proxy, method, args) -> {
            if (method.getName().equals("supportsNormalInitialization")) { return false; }
            if (method.getName().equals("randomNormal")) { throw new AssertionError("Loading must not initialize random weights"); }
            if (method.getName().equals("allocate") && count.incrementAndGet() == failAt) { throw new IllegalStateException("Injected parameter allocation failure"); }
            try { return method.invoke(cpu, args); } catch (InvocationTargetException error) { throw error.getCause(); }
        });
        return new Control(backend, count);
    }
    private static CausalLanguageModel.Builder decoder() {
        return CausalLanguageModel.builder().vocabularySize(7).modelWidth(8).blockCount(2).contextLength(8)
                .attention(a -> a.queryHeads(2).keyValueHeads(1)).feedForward(f -> f.hiddenWidth(12)).seed(42);
    }

    @Test void backendFilledConstantsAreTrainableLeavesWithOwnedGradients() {
        try (var context = ExecutionContext.cpu()) {
            try (var x = context.full(Shape.of(2, 3), .5f, true); var tape = context.gradientTape(); var sum = Ops.sum(x)) {
                assertTrue(x.isLeaf()); tape.backward(sum);
                try (var gradient = x.gradient()) { assertArrayEquals(new float[]{1, 1, 1, 1, 1, 1}, gradient.toFloatArray()); }
            }
            assertEquals(0, context.liveAllocations());
        }
    }

    @Test void explicitZeroLinearConstructionDoesNotRequireOrInvokeRandomInitialization() {
        var control = noRandom(Integer.MAX_VALUE);
        try (var context = ExecutionContext.create(control.backend)) {
            var definition = Linear.builder().inputFeatures(3).outputFeatures(2).define();
            assertThrows(UnsupportedOperationException.class, () -> definition.build(context)); assertEquals(0, control.allocations.get());
            try (var model = definition.build(context, ParameterInitialization.ZEROS)) {
                for (var parameter : model.namedParameters().values()) {
                    try (var tensor = parameter.value()) { assertTrue(tensor.isLeaf()); assertArrayEquals(new float[Math.toIntExact(tensor.shape().elements())], tensor.toFloatArray()); }
                }
            }
            assertEquals(0, context.liveAllocations());
        }
    }

    @Test void denseDefinitionsRestoreSuppliedWeightsWithoutChangingTheirArchitectureOrSeeds() {
        var definition = NeuralNetwork.builder().name("loaded").inputFeatures(3).dense("hidden", 4, Activation.Kind.SILU)
                .layerNorm("norm", 1e-5f).dropout("dropout", .1f).dense("out", 2).seed(17).define();
        var control = noRandom(Integer.MAX_VALUE);
        try (var cpu = ExecutionContext.cpu(); var context = ExecutionContext.create(control.backend); var original = definition.build(cpu);
             var loaded = definition.build(context, ParameterInitialization.ZEROS)) {
            original.training(false); StateDict.capture(original).applyTo(loaded);
            assertEquals(original.configuration(), loaded.configuration());
            try (var x = cpu.tensor(new float[]{1, 2, -1}, Shape.of(1, 3)); var y = context.tensor(new float[]{1, 2, -1}, x.shape());
                 var expected = original.forward(x); var actual = loaded.forward(y)) { assertArrayEquals(expected.toFloatArray(), actual.toFloatArray()); }
        }
    }

    @Test void transformerPoliciesPreserveTiesNormDefaultsAndAllProjectionWeights() {
        for (boolean tied : new boolean[]{false, true}) for (boolean plain : new boolean[]{false, true}) {
            var builder = decoder().tieInputOutputEmbeddings(tied).attention(a -> a.bias(plain));
            if (plain) { builder.normalization(NormalizationKind.LAYER).feedForward(f -> f.gelu()); }
            var definition = builder.define(); var control = noRandom(Integer.MAX_VALUE);
            try (var cpu = ExecutionContext.cpu(); var context = ExecutionContext.create(control.backend);
                 var original = definition.build(cpu); var loaded = definition.build(context, ParameterInitialization.ZEROS)) {
                for (var entry : loaded.namedParameters().entrySet()) {
                    try (var tensor = entry.getValue().value()) {
                        float[] expected = new float[Math.toIntExact(tensor.shape().elements())];
                        if (entry.getKey().contains("Norm.weight")) { java.util.Arrays.fill(expected, 1); }
                        assertArrayEquals(expected, tensor.toFloatArray(), entry.getKey()); assertTrue(tensor.isLeaf());
                    }
                }
                original.training(false); StateDict.capture(original).applyTo(loaded);
                assertEquals(tied, StateDict.capture(loaded).aliases().containsKey("output.weight"));
                try (var x = cpu.tensor(new int[]{1, 3, 2}, Shape.of(1, 3)); var y = context.tensor(new int[]{1, 3, 2}, x.shape());
                     var expected = original.forward(x); var actual = loaded.forward(y)) { assertArrayEquals(expected.toFloatArray(), actual.toFloatArray()); }
            }
        }
    }

    @Test void zeroInitializedTransformerReleasesEveryPartialConstructionAllocation() {
        var definition = decoder().tieInputOutputEmbeddings(false).normalization(NormalizationKind.LAYER).attention(a -> a.bias(true)).define();
        int allocations;
        var baseline = noRandom(Integer.MAX_VALUE);
        try (var context = ExecutionContext.create(baseline.backend); var model = definition.build(context, ParameterInitialization.ZEROS)) { allocations = baseline.allocations.get(); }
        for (int failAt = 1; failAt <= allocations; failAt++) {
            var control = noRandom(failAt);
            try (var context = ExecutionContext.create(control.backend)) {
                assertThrows(IllegalStateException.class, () -> definition.build(context, ParameterInitialization.ZEROS));
                context.synchronize(); assertEquals(0, context.liveAllocations(), "allocation " + failAt);
            }
        }
    }

    @Test void customLayersMustOptIntoAlternativeInitializationBeforeAnyAllocation() {
        LayerDefinition custom = new LayerDefinition() {
            public Shape outputShape(Shape shape) { return shape; }
            public void validate(ExecutionContext context, Shape shape, String path) { }
            public UnaryModule build(ExecutionContext context, Shape shape, long seed) { return new Activation(context, Activation.Kind.RELU); }
            public ComponentSpec specification(String name) { return new ComponentSpec("custom", 1, name, Map.of(), List.of()); }
        };
        var definition = NeuralNetwork.builder().inputFeatures(2).dense(2).layer("custom", custom).define(); var control = noRandom(Integer.MAX_VALUE);
        try (var context = ExecutionContext.create(control.backend)) {
            assertThrows(UnsupportedOperationException.class, () -> definition.build(context, ParameterInitialization.ZEROS)); assertEquals(0, control.allocations.get());
        }
    }
}

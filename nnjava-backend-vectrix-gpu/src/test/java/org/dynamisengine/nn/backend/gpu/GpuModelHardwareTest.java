package org.dynamisengine.nn.backend.gpu;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import org.dynamisengine.gpu.api.compute.GpuDevice;
import org.dynamisengine.gpu.api.compute.GpuProvider;
import org.dynamisengine.nn.formats.JsonDocuments;
import org.dynamisengine.nn.generation.GenerationSession;
import org.dynamisengine.nn.io.StateDict;
import org.dynamisengine.nn.nn.NeuralNetwork;
import org.dynamisengine.nn.nn.Activation;
import org.dynamisengine.nn.nn.NormalizationKind;
import org.dynamisengine.nn.nn.ParameterInitialization;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.pretrained.LlamaCheckpoint;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.transformer.CausalLanguageModel;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import static org.junit.jupiter.api.Assertions.*;

@EnabledIfSystemProperty(named = "nn.gpu.backend", matches = "METAL|VULKAN")
class GpuModelHardwareTest {
    private static GpuDevice device() throws Exception {
        var provider = (GpuProvider) Class.forName(System.getProperty("nn.gpu.backend").equals("METAL")
                ? "org.dynamisengine.gpu.metal.MetalProvider" : "org.dynamisengine.gpu.vulkan.VulkanProvider").getConstructor().newInstance();
        assertFalse(provider.adapters().isEmpty(), "Requested GPU is unavailable"); return provider.adapters().getFirst().open();
    }
    private static Path fixture(boolean tied) throws Exception {
        return Path.of(GpuModelHardwareTest.class.getResource("/pretrained/" + (tied ? "tied-llama" : "untied-llama")).toURI());
    }
    private static float[] reference(Path fixture) throws Exception {
        var rows = JsonDocuments.array(JsonDocuments.read(fixture.resolve("reference.json")).get("logits"), "logits");
        return rows.stream().flatMap(row -> ((java.util.List<?>) row).stream()).collect(java.util.stream.Collector.of(
                java.util.ArrayList<Float>::new, (values, item) -> values.add(((Number) item).floatValue()), (a, b) -> { a.addAll(b); return a; },
                values -> { float[] result = new float[values.size()]; for (int i = 0; i < result.length; i++) { result[i] = values.get(i); } return result; }));
    }
    private static void compare(float[] expected, float[] actual, GpuMode mode) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            if (mode == GpuMode.EXACT) { assertEquals(Float.floatToIntBits(expected[i]), Float.floatToIntBits(actual[i]), "bits " + i); }
            else { assertEquals(expected[i], actual[i], 3e-4f * Math.max(1, Math.abs(expected[i])), "value " + i); }
        }
    }
    private static void empty(ExecutionContext context, GpuBackend backend) {
        context.synchronize(); assertEquals(0, context.liveAllocations()); assertEquals(0, backend.liveBufferHandles());
    }

    @Test void localTiedAndUntiedCheckpointsMatchIndependentLogitsAndGrowingDeviceCaches() throws Exception {
        for (var mode : GpuMode.values()) for (boolean tied : new boolean[]{true, false}) {
            var backend = new GpuBackend(device(), mode); var checkpoint = LlamaCheckpoint.open(fixture(tied));
            try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(backend)) {
                try (var original = checkpoint.load(cpu, 16); var model = checkpoint.load(gpu, 16);
                     var input = gpu.tensor(new int[]{2, 5, 3, 7}, Shape.of(1, 4)); var cpuInput = cpu.tensor(new int[]{2, 5, 3, 7}, input.shape())) {
                    assertTrue(model.isTraining()); model.training(false); original.training(false);
                    try (var a = model.namedParameters().get("embedding.weight").value(); var b = model.namedParameters().get("output.weight").value()) { assertEquals(tied, a.id() == b.id()); }
                    float[] expected = reference(fixture(tied));
                    try (var logits = model.forward(input); var control = original.forward(cpuInput)) {
                        assertArrayEquals(expected, logits.toFloatArray(), 3e-5f); compare(control.toFloatArray(), logits.toFloatArray(), mode);
                    }
                    int before = gpu.liveAllocations();
                    try (var session = model.newSession(1); var prefix = Ops.slice(input, 1, 0, 2); var suffix = Ops.slice(input, 1, 2, 4);
                         var first = session.prefill(prefix); var second = session.decode(suffix)) {
                        assertArrayEquals(Arrays.copyOfRange(expected, 0, 38), first.toFloatArray(), 3e-5f);
                        assertArrayEquals(Arrays.copyOfRange(expected, 38, 76), second.toFloatArray(), 3e-5f); assertEquals(4, session.length());
                        session.reset(); try (var reset = session.prefill(prefix)) { compare(first.toFloatArray(), reset.toFloatArray(), mode); }
                    }
                    assertEquals(before, gpu.liveAllocations());
                    try (var a = GenerationSession.builder().model(original).maxNewTokens(3).sampling(s -> s.temperature(0)).build();
                         var b = GenerationSession.builder().model(model).maxNewTokens(3).sampling(s -> s.temperature(0)).build()) {
                        assertArrayEquals(a.generate(new int[]{2, 5}).tokenIds(), b.generate(new int[]{2, 5}).tokenIds());
                    }
                    assertEquals(before, gpu.liveAllocations());
                }
                empty(gpu, backend);
            }
        }
    }

    @Test void fluentLayerNormAndGeluDecodersRestoreSuppliedWeightsAndBatchedMasks() throws Exception {
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            var definition = CausalLanguageModel.builder().vocabularySize(7).modelWidth(8).blockCount(2).contextLength(8)
                    .attention(a -> a.queryHeads(2).keyValueHeads(2).bias(true)).feedForward(f -> f.hiddenWidth(12).gelu())
                    .normalization(NormalizationKind.LAYER).tieInputOutputEmbeddings(false).seed(17).define();
            try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(backend)) {
                try (var original = definition.build(cpu); var model = definition.build(gpu, ParameterInitialization.ZEROS);
                     var x = cpu.tensor(new int[]{1, 2, 3, 4, 2, 1}, Shape.of(2, 3)); var y = gpu.tensor(new int[]{1, 2, 3, 4, 2, 1}, x.shape());
                     var p = cpu.tensor(new boolean[]{true, true, true, true, false, false}, Shape.of(2, 1, 1, 3));
                     var q = gpu.tensor(new boolean[]{true, true, true, true, false, false}, p.shape())) {
                    original.training(false); StateDict.capture(original).applyTo(model);
                    try (var expected = original.forward(x, p); var actual = model.forward(y, q)) { compare(expected.toFloatArray(), actual.toFloatArray(), mode); }
                }
                empty(gpu, backend);
            }
        }
    }

    @Test void fluentDenseNetworkAndDefaultConstructionKeepExplicitProviderPolicies() throws Exception {
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            var definition = NeuralNetwork.builder().inputFeatures(3).dense("hidden", 4, Activation.Kind.SILU).rmsNorm("norm", 1e-5f).dense("out", 2).seed(11).define();
            try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(backend)) {
                assertThrows(UnsupportedOperationException.class, () -> definition.build(gpu)); assertEquals(0, gpu.liveAllocations());
                try (var original = definition.build(cpu); var model = definition.build(gpu, ParameterInitialization.ZEROS);
                     var x = cpu.tensor(new float[]{1, -2, .5f}, Shape.of(1, 3)); var y = gpu.tensor(new float[]{1, -2, .5f}, x.shape())) {
                    StateDict.capture(original).applyTo(model);
                    try (var expected = original.forward(x); var actual = model.forward(y)) { compare(expected.toFloatArray(), actual.toFloatArray(), mode); }
                }
                empty(gpu, backend);
            }
        }
    }

    @Test void deviceInferenceSessionsRejectStaleWeightsAndOverflowThenResetExplicitly() throws Exception {
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var context = ExecutionContext.create(backend)) {
                try (var model = LlamaCheckpoint.open(fixture(true)).load(context, 4); var input = context.tensor(new int[]{2, 5}, Shape.of(1, 2))) {
                    model.training(false); int before = context.liveAllocations();
                    try (var session = model.newSession(1)) {
                        try (var first = session.prefill(input); var second = session.decode(input)) { assertEquals(4, session.length()); }
                        assertThrows(IllegalArgumentException.class, () -> session.decode(input)); assertEquals(4, session.length());
                        session.reset(); try (var first = session.prefill(input); var weight = model.namedParameters().get("embedding.weight").value()) { context.fill(weight, 0); }
                        assertThrows(IllegalStateException.class, () -> session.decode(input));
                        session.reset(); try (var reset = session.prefill(input)) { assertArrayEquals(new float[38], reset.toFloatArray()); }
                    }
                    assertEquals(before, context.liveAllocations());
                }
                empty(context, backend);
            }
        }
    }

    @Test void failedCheckpointConstructionAndLateUploadsReleaseEveryOwnedParameter() throws Exception {
        for (var mode : GpuMode.values()) for (int failAt : new int[]{1, 10, 20, 21, 35, 40}) {
            var nativeDevice = device(); var count = new AtomicInteger();
            var wrapped = (GpuDevice) Proxy.newProxyInstance(getClass().getClassLoader(), new Class<?>[]{GpuDevice.class}, (proxy, method, args) -> {
                if (method.getName().equals("allocate") && args[0] instanceof org.dynamisengine.gpu.api.compute.BufferDescriptor descriptor
                        && descriptor.label().equals("nnjava") && count.incrementAndGet() == failAt) { throw new IllegalStateException("Injected checkpoint allocation failure"); }
                try { return method.invoke(nativeDevice, args); } catch (InvocationTargetException error) { throw error.getCause(); }
            });
            var backend = new GpuBackend(wrapped, mode);
            try (var context = ExecutionContext.create(backend)) {
                assertThrows(IllegalStateException.class, () -> LlamaCheckpoint.open(fixture(true)).load(context, 16)); empty(context, backend);
            }
        }
    }

    @Test void singletonBroadcastMatrixViewsUseValidLeadingDimensionsInBothModes() throws Exception {
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var context = ExecutionContext.create(backend)) {
                try (var vector = context.tensor(new float[]{1, 2, 3, 4}, Shape.of(4)); var row = Ops.broadcastTo(vector, Shape.of(1, 4));
                     var column = Ops.transpose(row); var dot = Ops.matmul(row, column); var outer = Ops.matmul(column, row)) {
                    assertEquals(0, row.layout().stride(0)); assertEquals(0, column.layout().stride(1));
                    assertArrayEquals(new float[]{30}, dot.toFloatArray());
                    assertArrayEquals(new float[]{1, 2, 3, 4, 2, 4, 6, 8, 3, 6, 9, 12, 4, 8, 12, 16}, outer.toFloatArray());
                }
                empty(context, backend);
            }
        }
    }
}

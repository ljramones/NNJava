package org.dynamisengine.nn.backend.gpu;

import java.util.Arrays;
import java.util.List;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.lang.reflect.Proxy;
import java.lang.reflect.InvocationTargetException;
import java.util.function.Function;
import org.dynamisengine.gpu.api.compute.GpuDevice;
import org.dynamisengine.gpu.api.compute.GpuProvider;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;
import org.dynamisengine.nn.nn.Sequential;
import org.dynamisengine.nn.nn.Parameter;
import org.dynamisengine.nn.nn.Linear;
import org.dynamisengine.nn.nn.Activation;
import org.dynamisengine.nn.nn.LayerNorm;
import org.dynamisengine.nn.nn.Dropout;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import static org.junit.jupiter.api.Assertions.*;

/** Numerical, gradient, layout and lifetime checks on the explicitly selected native runtime. */
@EnabledIfSystemProperty(named = "nn.gpu.backend", matches = "METAL|VULKAN")
class GpuNeuralHardwareTest {
    private static GpuDevice device() throws Exception {
        String platform = System.getProperty("nn.gpu.backend");
        var provider = (GpuProvider) Class.forName(platform.equals("METAL") ? "org.dynamisengine.gpu.metal.MetalProvider"
                : "org.dynamisengine.gpu.vulkan.VulkanProvider").getConstructor().newInstance();
        assertFalse(provider.adapters().isEmpty(), "Requested GPU is unavailable"); return provider.adapters().getFirst().open();
    }
    private record Derivative(float[] output, float[] input) { }
    private static Derivative derivative(ExecutionContext context, float[] values, Shape shape, boolean transpose,
                                          Function<Tensor, Tensor> operation) {
        try (var x = context.tensor(values, shape, true); var tape = context.gradientTape();
             var input = transpose ? Ops.transpose(x) : x.retain(); var y = operation.apply(input)) {
            float[] weights = new float[Math.toIntExact(y.shape().elements())];
            for (int i = 0; i < weights.length; i++) { weights[i] = (i % 7 - 3) * .2f; }
            try (var upstream = context.tensor(weights, y.shape())) { tape.backward(y, upstream); }
            try (var gradient = x.gradient()) { return new Derivative(y.toFloatArray(), gradient.toFloatArray()); }
        }
    }
    private static void compare(float[] expected, float[] actual, GpuMode mode, float tolerance) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            if (mode == GpuMode.EXACT || !Float.isFinite(expected[i])) {
                assertEquals(Float.floatToIntBits(expected[i]), Float.floatToIntBits(actual[i]), "bits at " + i);
            } else { assertEquals(expected[i], actual[i], tolerance * Math.max(1, Math.abs(expected[i])), "value at " + i); }
        }
    }
    private static void empty(ExecutionContext context, GpuBackend backend) {
        context.synchronize(); assertEquals(0, context.liveAllocations()); assertEquals(0, backend.liveBufferHandles());
    }

    @Test void activationOutputsAndInputGradientsMatchWithTransposeViews() throws Exception {
        List<Function<Tensor, Tensor>> functions = List.of(Ops::relu, Ops::sigmoid, Ops::tanh, Ops::gelu, Ops::silu);
        float[] values = {-8, -2, -.3f, 0, .125f, .75f, 2, 8};
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(backend)) {
                for (var function : functions) {
                    var expected = derivative(cpu, values, Shape.of(2, 4), true, function);
                    var actual = derivative(gpu, values, Shape.of(2, 4), true, function);
                    compare(expected.output, actual.output, mode, 3e-5f); compare(expected.input, actual.input, mode, 3e-5f); empty(gpu, backend);
                }
                assertEquals("vectrix-gpu", backend.executedTiers().get(Operation.GELU));
                assertTrue(backend.accuracyContracts().get(Operation.GELU).bound().kind().name().equals("EXACT"));
            }
        }
    }

    @Test void reluPreservesCpuNanInfinityAndSignedZeroPolicyInBothModes() throws Exception {
        float[] values = {Float.NaN, Float.NEGATIVE_INFINITY, -0f, 0f, Float.POSITIVE_INFINITY};
        for (var mode : GpuMode.values()) {
            try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(new GpuBackend(device(), mode));
                 var a = cpu.tensor(values, Shape.of(5)); var b = gpu.tensor(values, Shape.of(5)); var expected = Ops.relu(a); var actual = Ops.relu(b)) {
                compare(expected.toFloatArray(), actual.toFloatArray(), GpuMode.EXACT, 0);
            }
        }
    }

    @Test void softmaxAndLogSoftmaxHandleFullyMaskedRowsAndGradients() throws Exception {
        float[] values = {Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY,
                1, Float.NEGATIVE_INFINITY, 3, -2, 0, 2};
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(backend)) {
                for (boolean logarithmic : new boolean[]{false, true}) {
                    Function<Tensor, Tensor> operation = x -> logarithmic ? Ops.logSoftmax(x, -1) : Ops.softmax(x, -1);
                    var expected = derivative(cpu, values, Shape.of(3, 3), false, operation);
                    var actual = derivative(gpu, values, Shape.of(3, 3), false, operation);
                    compare(expected.output, actual.output, mode, 5e-5f); compare(expected.input, actual.input, mode, 5e-5f);
                    assertArrayEquals(new float[3], Arrays.copyOf(actual.input, 3)); empty(gpu, backend);
                }
                assertEquals("vectrix-gpu", backend.executedTiers().get(Operation.LOG_SOFTMAX));
            }
        }
    }

    @Test void nonLastAxisSoftmaxUsesDevicePackingAndRestoresGradientLayout() throws Exception {
        float[] values = {1, -2, .3f, 4, -.5f, 2};
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(backend)) {
                var expected = derivative(cpu, values, Shape.of(2, 3), false, x -> Ops.softmax(x, 0));
                var actual = derivative(gpu, values, Shape.of(2, 3), false, x -> Ops.softmax(x, 0));
                compare(expected.output, actual.output, mode, 5e-5f); compare(expected.input, actual.input, mode, 5e-5f); empty(gpu, backend);
            }
        }
    }
    @Test void counterBasedDropoutMatchesForwardMasksAndGradientsExactly() throws Exception {
        float[] values = new float[35]; for (int i = 0; i < values.length; i++) { values[i] = (i - 8) * .25f; }
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(backend)) {
                var expected = derivative(cpu, values, Shape.of(5, 7), true, x -> Ops.dropout(x, .3f, 19, 11));
                var actual = derivative(gpu, values, Shape.of(5, 7), true, x -> Ops.dropout(x, .3f, 19, 11));
                compare(expected.output, actual.output, GpuMode.EXACT, 0); compare(expected.input, actual.input, GpuMode.EXACT, 0); empty(gpu, backend);
                assertEquals("vectrix-gpu", backend.executedTiers().get(Operation.DROPOUT));
            }
        }
    }

    private record Normalized(float[] output, float[] x, float[] scale, float[] bias) { }
    private static Normalized normalized(ExecutionContext context, boolean rms, boolean noRows) {
        try (var x = context.tensor(noRows ? new float[0] : new float[]{1, -.2f, 3, -.4f, 5, 2}, Shape.of(noRows ? 0 : 2, 3), true);
             var scale = context.tensor(new float[]{1.2f, -.7f}, Shape.of(2), true);
             var bias = context.tensor(new float[]{.3f, .1f}, Shape.of(2), true); var tape = context.gradientTape();
             var input = noRows ? Ops.reshape(x, Shape.of(0, 2)) : Ops.transpose(x);
             var result = rms ? Ops.rmsNorm(input, scale, 1e-5f) : Ops.layerNorm(input, scale, bias, 1e-5f)) {
            float[] values = new float[Math.toIntExact(result.shape().elements())];
            for (int i = 0; i < values.length; i++) { values[i] = (i - 2) * .15f; }
            try (var upstream = context.tensor(values, result.shape())) { tape.backward(result, upstream); }
            try (var dx = x.gradient(); var ds = scale.gradient(); var db = rms ? null : bias.gradient()) {
                return new Normalized(result.toFloatArray(), dx.toFloatArray(), ds.toFloatArray(), db == null ? null : db.toFloatArray());
            }
        }
    }
    @Test void normalizationInputAndParameterGradientsMatchAcrossStridedRows() throws Exception {
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(backend)) {
                for (boolean rms : new boolean[]{false, true}) {
                    var expected = normalized(cpu, rms, false); var actual = normalized(gpu, rms, false);
                    compare(expected.output, actual.output, mode, 2e-4f); compare(expected.x, actual.x, mode, 2e-4f);
                    compare(expected.scale, actual.scale, mode, 2e-4f);
                    if (!rms) { compare(expected.bias, actual.bias, mode, 2e-4f); } empty(gpu, backend);
                }
            }
        }
    }
    @Test void emptyNormalizationRowsProduceZeroParameterGradients() throws Exception {
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(backend)) {
                for (boolean rms : new boolean[]{false, true}) {
                    var expected = normalized(cpu, rms, true); var actual = normalized(gpu, rms, true);
                    assertArrayEquals(expected.scale, actual.scale); assertArrayEquals(new float[0], actual.x);
                    if (!rms) { assertArrayEquals(expected.bias, actual.bias); } empty(gpu, backend);
                }
                try (var stored = gpu.tensor(new float[6], Shape.of(3, 2)); var tail = Ops.slice(stored, 0, 1, 3);
                     var view = Ops.slice(tail, 0, 1, 1); var scale = gpu.tensor(new float[]{1, 1}, Shape.of(2), true);
                     var bias = gpu.tensor(new float[]{0, 0}, Shape.of(2), true); var tape = gpu.gradientTape();
                     var output = Ops.layerNorm(view, scale, bias, 1e-5f); var seed = gpu.tensor(new float[0], Shape.of(0, 2))) {
                    tape.backward(output, seed);
                    try (var ds = scale.gradient(); var db = bias.gradient()) { assertArrayEquals(new float[2], ds.toFloatArray()); assertArrayEquals(new float[2], db.toFloatArray()); }
                }
                empty(gpu, backend);
            }
        }
    }
    @Test void normalizationParameterGradientsSpanMultipleUpstreamReductionBlocks() throws Exception {
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(backend)) {
                for (boolean rms : new boolean[]{false, true}) {
                    var expected = largeNorm(cpu, rms); var actual = largeNorm(gpu, rms);
                    compare(expected.output, actual.output, mode, 5e-4f); compare(expected.x, actual.x, mode, 5e-4f);
                    compare(expected.scale, actual.scale, mode, 5e-4f); if (!rms) { compare(expected.bias, actual.bias, mode, 5e-4f); }
                    empty(gpu, backend);
                }
            }
        }
    }
    private static Normalized largeNorm(ExecutionContext context, boolean rms) {
        float[] values = new float[259 * 17], upstream = new float[values.length], gamma = new float[17], beta = new float[17];
        for (int i = 0; i < values.length; i++) { values[i] = (i % 29 - 14) * .07f; upstream[i] = (i % 13 - 6) * .13f; }
        for (int i = 0; i < 17; i++) { gamma[i] = .7f + i * .02f; beta[i] = (i - 5) * .03f; }
        try (var x = context.tensor(values, Shape.of(259, 17), true); var g = context.tensor(gamma, Shape.of(17), true);
             var b = context.tensor(beta, Shape.of(17), true); var dy = context.tensor(upstream, Shape.of(259, 17));
             var tape = context.gradientTape(); var y = rms ? Ops.rmsNorm(x, g, 1e-5f) : Ops.layerNorm(x, g, b, 1e-5f)) {
            tape.backward(y, dy);
            try (var dx = x.gradient(); var dg = g.gradient(); var db = rms ? null : b.gradient()) {
                return new Normalized(y.toFloatArray(), dx.toFloatArray(), dg.toFloatArray(), db == null ? null : db.toFloatArray());
            }
        }
    }

    @Test void sumsSupportMultipleAxesViewsAndBackwardBroadcasting() throws Exception {
        float[] values = new float[24]; for (int i = 0; i < values.length; i++) { values[i] = (i - 9) * .125f; }
        List<Function<Tensor, Tensor>> operations = List.of(Ops::sum, x -> Ops.sum(x, new int[]{0, 2}, false),
                x -> Ops.sum(x, new int[]{0, 2}, true), x -> Ops.sumTo(x, Shape.of(1, 3, 1)));
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(backend)) {
                for (var operation : operations) {
                    var expected = derivative(cpu, values, Shape.of(2, 3, 4), false, operation);
                    var actual = derivative(gpu, values, Shape.of(2, 3, 4), false, operation);
                    compare(expected.output, actual.output, mode, 1e-5f); compare(expected.input, actual.input, mode, 1e-5f); empty(gpu, backend);
                }
            }
        }
    }
    @Test void exactSumToUsesCpuSerialOrderEvenWhenTreeSummationWouldDiffer() throws Exception {
        float[] values = {1e8f, 1, 1, -1e8f, 1, 1};
        try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(new GpuBackend(device(), GpuMode.EXACT));
             var a = cpu.tensor(values, Shape.of(6, 1)); var b = gpu.tensor(values, Shape.of(6, 1));
             var expected = Ops.sumTo(a, Shape.of(1)); var actual = Ops.sumTo(b, Shape.of(1))) {
            assertArrayEquals(new float[]{2}, expected.toFloatArray()); assertArrayEquals(expected.toFloatArray(), actual.toFloatArray());
        }
    }
    @Test void emptyReductionsAndRobustEuclideanNormHaveDefinedResults() throws Exception {
        for (var mode : GpuMode.values()) {
            try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(new GpuBackend(device(), mode));
                 var a = cpu.tensor(new float[]{1e30f, 1e-20f, -1e30f}, Shape.of(3)); var b = gpu.tensor(a.toFloatArray(), Shape.of(3));
                 var expected = Ops.norm(a); var actual = Ops.norm(b); var empty = gpu.tensor(new float[0], Shape.of(0, 3));
                 var sum = Ops.sum(empty); var axis = Ops.sum(empty, 0, false); var norm = Ops.norm(empty)) {
                assertArrayEquals(expected.toFloatArray(), actual.toFloatArray()); assertArrayEquals(new float[]{0}, sum.toFloatArray());
                assertArrayEquals(new float[3], axis.toFloatArray()); assertArrayEquals(new float[]{0}, norm.toFloatArray());
            }
        }
    }

    @Test void batchedBroadcastMatrixProductsAndGradientsMatchTheCpu() throws Exception {
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(backend)) {
                var expected = matrixGradient(cpu); var actual = matrixGradient(gpu);
                for (int i = 0; i < expected.length; i++) { compare(expected[i], actual[i], mode, 3e-5f); }
                empty(gpu, backend);
            }
        }
    }
    private static float[][] matrixGradient(ExecutionContext context) {
        float[] a = new float[24], b = new float[24];
        for (int i = 0; i < a.length; i++) { a[i] = (i - 7) * .1f; b[i] = (i % 9 - 3) * .2f; }
        try (var left = context.tensor(a, Shape.of(2, 1, 3, 4), true);
             var stored = context.tensor(b, Shape.of(1, 3, 2, 4), true);
             var tape = context.gradientTape(); var right = Ops.transpose(stored); var output = Ops.matmul(left, right); var loss = Ops.sum(output)) {
            tape.backward(loss);
            try (var da = left.gradient(); var db = stored.gradient()) { return new float[][]{output.toFloatArray(), da.toFloatArray(), db.toFloatArray()}; }
        }
    }
    @Test void unsupportedRowWidthsAndStorageComputeFailBeforeDispatch() throws Exception {
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var context = ExecutionContext.create(backend);
                 var x = context.tensor(new float[262145], Shape.of(1, 262145));
                 var half = context.tensor(new short[]{0x3c00}, DType.FP16, Shape.of(1))) {
                int baseline = backend.liveBufferHandles();
                assertThrows(UnsupportedOperationException.class, () -> Ops.softmax(x, -1));
                assertThrows(UnsupportedOperationException.class, () -> Ops.silu(half));
                assertEquals(baseline, backend.liveBufferHandles()); assertTrue(backend.executedTiers().isEmpty());
            }
        }
    }
    @Test void stridedMatrixOperandsAndZeroInnerDimensionsStayOnDevice() throws Exception {
        for (var mode : GpuMode.values()) {
            try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(new GpuBackend(device(), mode))) {
                float[] values = new float[24]; for (int i = 0; i < values.length; i++) { values[i] = i * .1f; }
                try (var c = cpu.tensor(values, Shape.of(4, 6)); var g = gpu.tensor(values, Shape.of(4, 6));
                     var cr = Ops.slice(c, 0, 1, 4, 2); var gr = Ops.slice(g, 0, 1, 4, 2);
                     var ca = Ops.slice(cr, 1, 1, 6, 2); var ga = Ops.slice(gr, 1, 1, 6, 2);
                     var cb = cpu.tensor(new float[]{1, 2, 3, 4, 5, 6}, Shape.of(3, 2)); var gb = gpu.tensor(cb.toFloatArray(), cb.shape());
                     var expected = Ops.matmul(ca, cb); var actual = Ops.matmul(ga, gb);
                     var emptyA = gpu.tensor(new float[0], Shape.of(2, 3, 0)); var emptyB = gpu.tensor(new float[0], Shape.of(0, 4));
                     var zero = Ops.matmul(emptyA, emptyB)) {
                    compare(expected.toFloatArray(), actual.toFloatArray(), mode, 2e-5f); assertArrayEquals(new float[24], zero.toFloatArray());
                }
            }
        }
    }
    private record Trained(float[] losses, Map<String, float[]> parameters) { }
    @Test void publicDenseLayersNormAndDropoutComposeForSeveralGpuGradientUpdates() throws Exception {
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(backend)) {
                var expected = train(cpu); var actual = train(gpu);
                compare(expected.losses, actual.losses, mode, 3e-4f);
                expected.parameters.forEach((name, weights) -> compare(weights, actual.parameters.get(name), mode, 5e-4f));
                empty(gpu, backend);
            }
        }
    }
    private static Trained train(ExecutionContext context) {
        try (var w1 = context.tensor(new float[]{.1f, .2f, -.3f, -.4f, .5f, .2f, .6f, -.1f, .3f, -.3f, .2f, .4f}, Shape.of(4, 3), true);
             var w2 = context.tensor(new float[]{.2f, -.1f, .3f, .1f, -.3f, .2f, .4f, -.2f}, Shape.of(2, 4), true);
             var p1 = new Parameter("first", w1); var p2 = new Parameter("last", w2);
             var model = new Sequential(context, new Linear(context, p1, true), Activation.silu(context), new LayerNorm(context, 4, 1e-5f),
                     new Dropout(context, .2f, 42), new Linear(context, p2, true));
             var x = context.tensor(new float[]{1, 0, -.5f, 0, 1, .5f, -.5f, .5f, 1}, Shape.of(3, 3));
             var target = context.tensor(new float[]{.2f, -.1f, -.3f, .1f, .4f, -.2f}, Shape.of(3, 2))) {
            float[] losses = new float[3];
            for (int step = 0; step < 3; step++) {
                model.namedParameters().values().forEach(Parameter::clearGradient);
                try (var tape = context.gradientTape(); var prediction = model.forward(x); var difference = Ops.subtract(prediction, target);
                     var squared = Ops.square(difference); var loss = Ops.mean(squared)) {
                    losses[step] = loss.toFloatArray()[0]; tape.backward(loss);
                }
                for (var parameter : model.namedParameters().values()) {
                    try (var value = parameter.value(); var gradient = value.gradient()) { context.axpy(value, gradient, -.01f); }
                }
            }
            var weights = new LinkedHashMap<String, float[]>();
            model.namedParameters().forEach((name, parameter) -> { try (var tensor = parameter.value()) { weights.put(name, tensor.toFloatArray()); } });
            return new Trained(losses, weights);
        }
    }
    @Test void abortedRecordingsReleaseScratchHandlesAtEveryAllocationBoundary() throws Exception {
        for (var mode : GpuMode.values()) for (int failure = 1; failure <= 6; failure++) {
            var nativeDevice = device(); var allocations = new AtomicInteger(); var failAt = new AtomicInteger(Integer.MAX_VALUE);
            var wrapped = (GpuDevice) Proxy.newProxyInstance(getClass().getClassLoader(), new Class<?>[]{GpuDevice.class}, (proxy, method, args) -> {
                if (method.getName().equals("allocate") && args[0] instanceof org.dynamisengine.gpu.api.compute.BufferDescriptor descriptor
                        && descriptor.label().equals("nnjava") && allocations.incrementAndGet() == failAt.get()) { throw new IllegalStateException("Injected GPU scratch allocation failure"); }
                try { return method.invoke(nativeDevice, args); } catch (InvocationTargetException error) { throw error.getCause(); }
            });
            var backend = new GpuBackend(wrapped, mode); var context = ExecutionContext.create(backend);
            try {
                try (var x = context.tensor(new float[]{1, 2, 3, 4, 5, 6}, Shape.of(2, 3)); var view = Ops.transpose(x)) {
                    context.synchronize(); int before = backend.liveBufferHandles(); failAt.set(allocations.get() + failure + 1); // The public output precedes scratch allocations.
                    assertThrows(IllegalStateException.class, () -> Ops.softmax(view, -1));
                    assertEquals(before, backend.liveBufferHandles());
                }
                assertEquals(0, context.liveAllocations());
            } finally { assertThrows(IllegalStateException.class, context::close); }
            assertEquals(0, backend.liveBufferHandles());
        }
    }
}

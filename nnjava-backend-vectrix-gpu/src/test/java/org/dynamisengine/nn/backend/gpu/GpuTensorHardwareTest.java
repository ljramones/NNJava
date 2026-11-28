package org.dynamisengine.nn.backend.gpu;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import org.dynamisengine.gpu.api.compute.ComputeException;
import org.dynamisengine.gpu.api.compute.GpuDevice;
import org.dynamisengine.gpu.api.compute.GpuProvider;
import org.dynamisengine.nn.backend.KernelAttributes;
import org.dynamisengine.nn.backend.KernelCall;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;
import org.dynamisengine.nn.tensor.TensorAccess;
import org.dynamisengine.nn.transformer.Attention;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import static org.junit.jupiter.api.Assertions.*;

/** Native contracts for tensor programs used by embeddings, attention and recurrent state selection. */
@EnabledIfSystemProperty(named = "nn.gpu.backend", matches = "METAL|VULKAN")
class GpuTensorHardwareTest {
    private static GpuDevice device() throws Exception {
        var provider = (GpuProvider) Class.forName(System.getProperty("nn.gpu.backend").equals("METAL")
                ? "org.dynamisengine.gpu.metal.MetalProvider" : "org.dynamisengine.gpu.vulkan.VulkanProvider").getConstructor().newInstance();
        assertFalse(provider.adapters().isEmpty(), "Requested GPU is unavailable"); return provider.adapters().getFirst().open();
    }
    private record Snapshot(Shape shape, DType dtype, List<Long> words) { }
    private static Snapshot snapshot(Tensor tensor) {
        try (tensor) {
            var words = new ArrayList<Long>();
            switch (tensor.dtype()) {
                case FP32 -> { for (float value : tensor.toFloatArray()) { words.add((long) Float.floatToIntBits(value)); } }
                case INT32 -> { for (int value : tensor.toIntArray()) { words.add((long) value); } }
                case BOOL -> { for (boolean value : tensor.toBooleanArray()) { words.add(value ? 1L : 0L); } }
                default -> throw new AssertionError("Unexpected fixture type");
            }
            return new Snapshot(tensor.shape(), tensor.dtype(), List.copyOf(words));
        }
    }
    private static void empty(ExecutionContext context, GpuBackend backend) {
        context.synchronize(); assertEquals(0, context.liveAllocations()); assertEquals(0, backend.liveBufferHandles());
    }
    private static void bits(float[] expected, float[] actual) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) { assertEquals(Float.floatToIntBits(expected[i]), Float.floatToIntBits(actual[i]), "bits at " + i); }
    }
    private static void near(float[] expected, float[] actual, GpuMode mode) {
        if (mode == GpuMode.EXACT) { bits(expected, actual); return; }
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) { assertEquals(expected[i], actual[i], 3e-4f * Math.max(1, Math.abs(expected[i])), "value at " + i); }
    }
    private static float[] values(int size) {
        float[] result = new float[size]; for (int i = 0; i < size; i++) { result[i] = (i % 19 - 9) * .125f; } return result;
    }

    private static List<Snapshot> logical(ExecutionContext context) {
        var result = new ArrayList<Snapshot>();
        try (var store = context.tensor(new float[]{Float.NaN, -0f, 0f, Float.MIN_VALUE, -Float.MIN_VALUE, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, 2}, Shape.of(2, 4));
             var x = Ops.transpose(store); var zero = context.full(Shape.of(), 0);
             var ints = context.tensor(new int[]{Integer.MIN_VALUE, -16777217, 16777217, Integer.MAX_VALUE}, Shape.of(4, 1));
             var pivot = context.tensor(new int[]{-16777216, 16777216}, Shape.of(2));
             var p = context.tensor(new boolean[]{true, false, true, false}, Shape.of(4, 1));
             var q = context.tensor(new boolean[]{false, true}, Shape.of(2))) {
            result.add(snapshot(Ops.equal(x, zero))); result.add(snapshot(Ops.less(x, zero))); result.add(snapshot(Ops.greater(x, zero)));
            result.add(snapshot(Ops.isFinite(x))); result.add(snapshot(Ops.equal(ints, pivot)));
            result.add(snapshot(Ops.less(ints, pivot))); result.add(snapshot(Ops.greater(ints, pivot)));
            result.add(snapshot(Ops.and(p, q))); result.add(snapshot(Ops.or(p, q))); result.add(snapshot(Ops.not(p)));
            result.add(snapshot(Ops.less(p, q))); result.add(snapshot(Ops.where(p, ints, pivot)));
            result.add(snapshot(Ops.where(p, x, zero))); result.add(snapshot(Ops.where(p, p, q)));
        }
        return result;
    }
    @Test void comparisonsLogicalSelectionAndClassificationRetainWordSemantics() throws Exception {
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(backend)) {
                assertEquals(logical(cpu), logical(gpu));
                assertEquals("EXACT", backend.accuracyContracts().get(Operation.WHERE).bound().kind().name()); empty(gpu, backend);
            }
        }
    }

    private static List<Snapshot> casts(ExecutionContext context) {
        var result = new ArrayList<Snapshot>();
        try (var floats = context.tensor(new float[]{Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, -0f, .5f, -1.9f, 1e30f, -1e30f}, Shape.of(2, 4));
             var ints = context.tensor(new int[]{Integer.MIN_VALUE, Integer.MAX_VALUE, -16777217, 16777217, 0, 1, -1, 3}, Shape.of(2, 4));
             var bools = context.tensor(new boolean[]{false, true, false, true, true, false, true, false}, Shape.of(2, 4))) {
            for (var stored : List.of(floats, ints, bools)) {
                try (var transposed = Ops.transpose(stored)) {
                    for (var dtype : List.of(DType.FP32, DType.INT32, DType.BOOL)) { result.add(snapshot(Ops.cast(transposed, dtype))); }
                }
            }
        }
        return result;
    }
    @Test void castsMatchJavaSaturationAndLargeIntegerRoundingAcrossStridedInputs() throws Exception {
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(backend)) { assertEquals(casts(cpu), casts(gpu)); empty(gpu, backend); }
        }
    }

    private static List<Snapshot> highRank(ExecutionContext context) {
        var result = new ArrayList<Snapshot>();
        try (var p = context.tensor(new boolean[]{true, false, true}, Shape.of(1, 1, 3, 1, 1));
             var q = context.tensor(new boolean[]{true, false}, Shape.of(2, 1, 1, 1, 1));
             var both = Ops.and(p, q); var either = Ops.or(p, q); var vacuous = context.tensor(new boolean[0], Shape.of(2, 0, 3));
             var allTrue = context.tensor(new boolean[]{true, true}, Shape.of(2)); var scalar = context.tensor(new boolean[]{true}, Shape.of())) {
            result.add(snapshot(both.retain())); result.add(snapshot(Ops.not(either))); result.add(snapshot(Ops.all(both)));
            result.add(snapshot(Ops.all(vacuous))); result.add(snapshot(Ops.all(allTrue))); result.add(snapshot(Ops.all(scalar)));
        }
        return result;
    }
    @Test void highRankBroadcastMasksAndAllIncludeTheEmptyIdentity() throws Exception {
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(backend)) { assertEquals(highRank(cpu), highRank(gpu)); empty(gpu, backend); }
        }
    }

    private record Derivative(float[] output, float[] gradient) { }
    private static Derivative derivative(ExecutionContext context, float[] data, Shape shape, Function<Tensor, Tensor> function) {
        try (var x = context.tensor(data, shape, true); var tape = context.gradientTape(); var y = function.apply(x);
             var seed = context.tensor(values(Math.toIntExact(y.shape().elements())), y.shape())) {
            tape.backward(y, seed); try (var dx = x.gradient()) { return new Derivative(y.toFloatArray(), dx.toFloatArray()); }
        }
    }
    private static Derivative gather(ExecutionContext context) {
        return derivative(context, values(120), Shape.of(2, 3, 4, 5), x -> {
            try (var transposed = Ops.permute(x, 0, 2, 1, 3);
                 var indexStore = context.tensor(new int[]{2, 99, 0, 99, 2, 99, 1, 99}, Shape.of(2, 4));
                 var ids = Ops.slice(indexStore, 1, 0, 4, 2)) { return Ops.gather(transposed, 2, ids); }
        });
    }
    @Test void multidimensionalRepeatedGatherIndicesProduceExactEmbeddingGradients() throws Exception {
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(backend)) {
                var expected = gather(cpu); var actual = gather(gpu); bits(expected.output, actual.output); bits(expected.gradient, actual.gradient); empty(gpu, backend);
            }
        }
    }

    private static List<Snapshot> indexedWords(ExecutionContext context) {
        var result = new ArrayList<Snapshot>();
        try (var integers = context.tensor(new int[]{7, 8, 9, 10, 11, 12}, Shape.of(2, 3));
             var booleans = context.tensor(new boolean[]{true, false, true, false, true, false}, Shape.of(2, 3));
             var scalar = context.tensor(new int[]{1}, Shape.of()); var ids = context.tensor(new int[]{2, 0}, Shape.of(1, 2));
             var none = context.tensor(new int[0], Shape.of(0)); var empty = context.tensor(new float[0], Shape.of(2, 0))) {
            result.add(snapshot(Ops.gather(integers, 0, scalar))); result.add(snapshot(Ops.gather(booleans, 1, ids)));
            result.add(snapshot(Ops.gather(integers, 1, none))); result.add(snapshot(Ops.scatterAdd(empty, 1, none, Shape.of(2, 3))));
        }
        return result;
    }
    @Test void scalarAndEmptyIndicesWorkForIntegerBooleanAndFloatingPointTensors() throws Exception {
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(backend)) { assertEquals(indexedWords(cpu), indexedWords(gpu)); empty(gpu, backend); }
        }
    }

    @Test void scatterAddPreservesSerialCancellationOrderAndItsBackwardGather() throws Exception {
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(backend)) {
                Function<ExecutionContext, Derivative> fixture = context -> derivative(context, new float[]{1e8f, 1, 1, -1e8f, 1, 1}, Shape.of(6), x -> {
                    try (var ids = context.tensor(new int[6], Shape.of(6))) { return Ops.scatterAdd(x, 0, ids, Shape.of(2)); }
                });
                var expected = fixture.apply(cpu); var actual = fixture.apply(gpu);
                assertArrayEquals(new float[]{2, 0}, actual.output); bits(expected.output, actual.output); bits(expected.gradient, actual.gradient); empty(gpu, backend);
            }
        }
    }

    private static Derivative slicedConcat(ExecutionContext context) {
        return derivative(context, values(24), Shape.of(4, 6), x -> {
            try (var transposed = Ops.transpose(x); var a = Ops.slice(transposed, 0, 1, 6, 2); var b = Ops.slice(transposed, 0, 0, 6, 2);
                 var zero = Ops.slice(transposed, 0, 0, 0)) { return Ops.concat(0, zero, a, b); }
        });
    }
    @Test void concatAndSteppedSliceGradientsRestoreTheOriginalTensorLayout() throws Exception {
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(backend)) {
                var expected = slicedConcat(cpu); var actual = slicedConcat(gpu); bits(expected.output, actual.output); bits(expected.gradient, actual.gradient); empty(gpu, backend);
            }
        }
    }

    @Test void integerAndBooleanConcatNormalizeNegativeKernelAxes() throws Exception {
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var context = ExecutionContext.create(backend)) {
                try (var left = context.tensor(new int[]{1, 2, 3, 4}, Shape.of(2, 2)); var right = context.tensor(new int[]{7, 8}, Shape.of(2, 1));
                     var output = context.empty(DType.INT32, Shape.of(2, 3));
                     var a = context.tensor(new boolean[]{true, false, false, true}, Shape.of(2, 2));
                     var b = context.tensor(new boolean[]{true, false}, Shape.of(2, 1)); var joined = Ops.concat(-1, a, b)) {
                    context.synchronize();
                    backend.execute(new KernelCall(Operation.CONCAT, List.of(TensorAccess.kernel(left), TensorAccess.kernel(right)),
                            TensorAccess.kernel(output), 0, KernelAttributes.of("axis", -1))).await();
                    assertArrayEquals(new int[]{1, 2, 7, 3, 4, 8}, output.toIntArray());
                    assertArrayEquals(new boolean[]{true, false, true, false, true, false}, joined.toBooleanArray());
                }
                empty(context, backend);
            }
        }
    }

    private static List<Snapshot> maxima(ExecutionContext context) {
        var result = new ArrayList<Snapshot>();
        try (var stored = context.tensor(new float[]{-0f, 0f, -1, 0f, -0f, -1, 3, 3, 2, Float.NaN, Float.NaN, 7, 4, Float.NaN, 4, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY}, Shape.of(6, 3));
             var x = Ops.transpose(stored)) {
            result.add(snapshot(Ops.max(x, 0, true))); result.add(snapshot(Ops.argmax(x, 0, false)));
        }
        return result;
    }
    @Test void maxAndArgmaxChooseTheFirstTieOrNanIncludingSignedZero() throws Exception {
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(backend)) {
                assertEquals(maxima(cpu), maxima(gpu));
                Function<ExecutionContext, Derivative> fixture = context -> derivative(context, new float[]{2, 2, 1, 1, Float.NaN, Float.NaN}, Shape.of(2, 3), x -> Ops.max(x, 1, true));
                var expected = fixture.apply(cpu); var actual = fixture.apply(gpu); bits(expected.output, actual.output); bits(expected.gradient, actual.gradient); empty(gpu, backend);
            }
        }
    }

    @Test void multiAxisMaxBackwardRestoresTheOriginalOrderFromPackedReductionRows() throws Exception {
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(backend)) {
                var expected = derivative(cpu, values(24), Shape.of(4, 3, 2), x -> {
                    try (var permuted = Ops.permute(x, 1, 2, 0); var packed = Ops.contiguous(permuted); var matrix = Ops.reshape(packed, Shape.of(3, 8))) {
                        return Ops.max(matrix, 1, false);
                    }
                });
                try (var stored = gpu.tensor(values(24), Shape.of(4, 3, 2)); var input = Ops.permute(stored, 2, 1, 0);
                     var output = gpu.empty(DType.FP32, Shape.of(1, 3, 1)); var seed = gpu.tensor(values(3), output.shape());
                     var dx = gpu.empty(DType.FP32, input.shape())) {
                    gpu.synchronize(); var attributes = KernelAttributes.of("axes", List.of(-1, 0)).and("keep", true);
                    backend.execute(new KernelCall(Operation.REDUCE_MAX, List.of(TensorAccess.kernel(input)), TensorAccess.kernel(output), 0, attributes)).await();
                    backend.execute(new KernelCall(Operation.MAX_BACKWARD, List.of(TensorAccess.kernel(input), TensorAccess.kernel(seed)), TensorAccess.kernel(dx), 0, attributes)).await();
                    bits(expected.output, output.toFloatArray()); try (var restored = Ops.permute(dx, 2, 1, 0)) { bits(expected.gradient, restored.toFloatArray()); }
                }
                empty(gpu, backend);
            }
        }
    }

    private static Derivative rotary(ExecutionContext context, boolean half, boolean explicit) {
        return derivative(context, values(192), Shape.of(2, 3, 8, 4), x -> {
            try (var input = Ops.permute(x, 0, 1, 3, 2);
                 var stored = context.tensor(new int[]{0, 99, 1, 99, 20000001, 99, Integer.MAX_VALUE, 99, 2, 99, 17, 99, 5000003, 99, 100, 99}, Shape.of(2, 8));
                 var sliced = Ops.slice(stored, 1, 0, 8, 2); var packed = Ops.contiguous(sliced); var positions = Ops.reshape(packed, Shape.of(2, 1, 4))) {
                return explicit ? Ops.rope(input, positions, 10000, half) : Ops.rope(input, (1L << 40) + 17, 10000, half);
            }
        });
    }
    @Test void rotaryForwardAndBackwardMatchCpuAtLongOffsetsAndBroadcastPositions() throws Exception {
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(backend)) {
                for (boolean half : new boolean[]{false, true}) for (boolean explicit : new boolean[]{false, true}) {
                    var expected = rotary(cpu, half, explicit); var actual = rotary(gpu, half, explicit);
                    bits(expected.output, actual.output); bits(expected.gradient, actual.gradient); empty(gpu, backend);
                }
                assertEquals("vectrix-gpu", backend.executedTiers().get(Operation.ROPE));
            }
        }
    }

    @Test void scalarPositionsAndEmptyRotarySequencesHaveDefinedResults() throws Exception {
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(backend)) {
                Function<ExecutionContext, List<Snapshot>> fixture = context -> {
                    try (var x = context.tensor(values(16), Shape.of(2, 2, 4)); var position = context.tensor(new int[]{13}, Shape.of());
                         var none = context.tensor(new float[0], Shape.of(2, 0, 4))) {
                        return List.of(snapshot(Ops.rope(x, position, 10000, false)), snapshot(Ops.rope(none, 4, 10000, true)));
                    }
                };
                assertEquals(fixture.apply(cpu), fixture.apply(gpu)); empty(gpu, backend);
            }
        }
    }

    private record AttentionResult(float[] output, float[] embeddingGradient, float[] valueGradient) { }
    private static AttentionResult attention(ExecutionContext context) {
        try (var table = context.tensor(values(40), Shape.of(5, 8), true); var value = context.tensor(values(9), Shape.of(1, 1, 3, 3), true);
             var tokens = context.tensor(new int[]{0, 1, 0}, Shape.of(1, 3)); var positions = context.tensor(new int[]{2, 4, 7}, Shape.of(1, 1, 3));
             var allowed = context.tensor(new boolean[]{true, true, true, true, true, false, false, false, false}, Shape.of(1, 1, 3, 3));
             var tape = context.gradientTape(); var embedding = Ops.gather(table, 0, tokens); var qs = Ops.reshape(embedding, Shape.of(1, 3, 2, 4));
             var q = Ops.permute(qs, 0, 2, 1, 3); var ks = Ops.slice(embedding, 2, 0, 4); var kp = Ops.contiguous(ks); var k = Ops.reshape(kp, Shape.of(1, 1, 3, 4));
             var qr = Ops.rope(q, positions, 10000, true); var kr = Ops.rope(k, positions, 10000, true);
             var output = Attention.scaledDotProduct(qr, kr, value, allowed, true);
             var seed = context.tensor(values(18), output.shape())) {
            tape.backward(output, seed);
            try (var dt = table.gradient(); var dv = value.gradient()) { return new AttentionResult(output.toFloatArray(), dt.toFloatArray(), dv.toFloatArray()); }
        }
    }
    @Test void embeddingsRotaryAndMaskedGroupedQueryAttentionComposeWithGradients() throws Exception {
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var cpu = ExecutionContext.cpu(); var gpu = ExecutionContext.create(backend)) {
                var expected = attention(cpu); var actual = attention(gpu);
                near(expected.output, actual.output, mode); near(expected.embeddingGradient, actual.embeddingGradient, mode); near(expected.valueGradient, actual.valueGradient, mode);
                assertArrayEquals(new float[3], Arrays.copyOfRange(actual.output, 6, 9));
                assertArrayEquals(new float[3], Arrays.copyOfRange(actual.output, 15, 18)); empty(gpu, backend);
            }
        }
    }

    private static boolean kernelError(Throwable error) {
        if (error instanceof ComputeException compute && compute.code() == ComputeException.Code.KERNEL_ERROR) { return true; }
        if (error.getCause() != null && kernelError(error.getCause())) { return true; }
        for (var suppressed : error.getSuppressed()) { if (kernelError(suppressed)) { return true; } } return false;
    }
    @Test void badIndicesAndNegativePositionsFailThroughCompletionIncludingEmptyDomains() throws Exception {
        for (var mode : GpuMode.values()) for (int scenario = 0; scenario < 5; scenario++) {
            var backend = new GpuBackend(device(), mode); var context = ExecutionContext.create(backend); int kind = scenario;
            try {
                try (var x = context.tensor(values(8), Shape.of(2, 4)); var ids = context.tensor(new int[]{kind == 0 ? 2 : -1}, Shape.of(1));
                     var none = context.tensor(new float[0], Shape.of(kind == 3 ? 1 : 2, 0))) {
                    context.synchronize();
                    var failure = assertThrows(RuntimeException.class, () -> {
                        try (var result = switch (kind) {
                            case 0, 1 -> Ops.gather(x, 0, ids);
                            case 2 -> Ops.gather(none, 0, ids);
                            case 3 -> Ops.scatterAdd(none, 0, ids, Shape.of(2, 0));
                            default -> Ops.rope(x, ids, 10000, false);
                        }) { context.synchronize(); }
                    });
                    assertTrue(kernelError(failure), "Expected a reported GPU index error: " + failure);
                }
            } finally {
                assertThrows(RuntimeException.class, context::close); assertEquals(0, context.liveAllocations()); assertEquals(0, backend.liveBufferHandles());
            }
        }
    }

    @Test void unsupportedLowPrecisionTensorProgramsRejectBeforeDispatch() throws Exception {
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var context = ExecutionContext.create(backend); var half = context.tensor(new short[]{0, 0}, DType.FP16, Shape.of(2));
                 var ids = context.tensor(new int[]{0}, Shape.of(1))) {
                context.synchronize(); int before = backend.liveBufferHandles();
                assertThrows(UnsupportedOperationException.class, () -> Ops.cast(half, DType.FP32));
                assertThrows(UnsupportedOperationException.class, () -> Ops.gather(half, 0, ids));
                assertEquals(before, backend.liveBufferHandles()); assertTrue(backend.executedTiers().isEmpty());
            }
        }
    }

    @Test void rotaryAllocationFailuresReleaseUploadedPlansPositionsAndAbortedRecordings() throws Exception {
        for (var mode : GpuMode.values()) for (int failure = 1; failure <= 7; failure++) {
            var nativeDevice = device(); var allocations = new AtomicInteger(); var failAt = new AtomicInteger(Integer.MAX_VALUE);
            var wrapped = (GpuDevice) Proxy.newProxyInstance(getClass().getClassLoader(), new Class<?>[]{GpuDevice.class}, (proxy, method, args) -> {
                if (method.getName().equals("allocate") && args[0] instanceof org.dynamisengine.gpu.api.compute.BufferDescriptor descriptor
                        && descriptor.label().equals("nnjava") && allocations.incrementAndGet() == failAt.get()) { throw new IllegalStateException("Injected rotary allocation failure"); }
                try { return method.invoke(nativeDevice, args); } catch (InvocationTargetException error) { throw error.getCause(); }
            });
            var backend = new GpuBackend(wrapped, mode); var context = ExecutionContext.create(backend);
            try {
                try (var input = context.tensor(values(48), Shape.of(2, 2, 3, 4)); var positions = context.tensor(new int[]{1, 3, 5}, Shape.of(3))) {
                    context.synchronize(); int before = backend.liveBufferHandles(); failAt.set(allocations.get() + failure + 1);
                    assertThrows(IllegalStateException.class, () -> Ops.rope(input, positions, 10000, true));
                    assertEquals(before, backend.liveBufferHandles());
                }
                assertEquals(0, context.liveAllocations());
            } finally { assertThrows(IllegalStateException.class, context::close); }
            assertEquals(0, backend.liveBufferHandles());
        }
    }
}

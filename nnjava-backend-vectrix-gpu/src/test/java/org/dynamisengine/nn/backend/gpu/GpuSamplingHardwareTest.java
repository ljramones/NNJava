package org.dynamisengine.nn.backend.gpu;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.dynamisengine.gpu.api.compute.ComputeException;
import org.dynamisengine.gpu.api.compute.GpuDevice;
import org.dynamisengine.gpu.api.compute.GpuProvider;
import org.dynamisengine.gpu.primitives.Sampler;
import org.dynamisengine.nn.backend.Backend;
import org.dynamisengine.nn.backend.KernelTensor;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.generation.DeviceGenerationSession;
import org.dynamisengine.nn.generation.GenerationSession;
import org.dynamisengine.nn.generation.TensorAutoregressiveModel;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.ops.SamplingOptions;
import org.dynamisengine.nn.pretrained.LlamaCheckpoint;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import static org.junit.jupiter.api.Assertions.*;

@EnabledIfSystemProperty(named = "nn.gpu.backend", matches = "METAL|VULKAN")
class GpuSamplingHardwareTest {
    private static GpuDevice device() throws Exception {
        var provider = (GpuProvider) Class.forName(System.getProperty("nn.gpu.backend").equals("METAL")
                ? "org.dynamisengine.gpu.metal.MetalProvider" : "org.dynamisengine.gpu.vulkan.VulkanProvider").getConstructor().newInstance();
        assertFalse(provider.adapters().isEmpty(), "Requested GPU is unavailable"); return provider.adapters().getFirst().open();
    }
    private static void empty(ExecutionContext context, GpuBackend backend) {
        context.synchronize(); assertEquals(0, context.liveAllocations()); assertEquals(0, backend.liveBufferHandles());
    }

    @Test void greedyRowsPreserveFirstTiesSubnormalsAndStridedOffsetLayouts() throws Exception {
        float[][] rows = {{-0f, 0f, -1, -2, Float.MIN_VALUE}, {Float.MIN_VALUE, 0, -Float.MIN_VALUE, Float.NEGATIVE_INFINITY, Float.MIN_VALUE}, {3, 3, Float.NEGATIVE_INFINITY, -1, -2}};
        float[] stored = new float[18];
        for (int column = 0; column < 5; column++) for (int row = 0; row < 3; row++) { stored[3 + column * 3 + row] = rows[row][column]; }
        int[] expected = Arrays.stream(rows).mapToInt(values -> org.dynamisengine.vectrix.nn.Sampler.argmax(values, 0, values.length)).toArray();
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var context = ExecutionContext.create(backend)) {
                try (var storage = context.tensor(stored, Shape.of(6, 3)); var tail = Ops.slice(storage, 0, 1, 6); var logits = Ops.transpose(tail);
                     var selected = Ops.sample(logits, SamplingOptions.greedy(), 0)) {
                    assertArrayEquals(expected, selected.toIntArray()); assertEquals(Shape.of(3), selected.shape());
                    assertEquals("vectrix-gpu", backend.executedTiers().get(Operation.SAMPLE));
                }
                empty(context, backend);
            }
        }
    }

    @Test void stochasticSamplingMatchesPublishedReferenceAndResetsFullWidthSeedAndStep() throws Exception {
        int width = 2053, rows = 3; float[] logits = new float[rows * width];
        for (int i = 0; i < logits.length; i++) { logits[i] = (i % 79 - 39) * .125f; }
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var context = ExecutionContext.create(backend)) {
                try (var input = context.tensor(logits, Shape.of(rows, width))) {
                    if (mode == GpuMode.EXACT) {
                        var options = new SamplingOptions(.7f, 17, .85f, 42);
                        assertFalse(context.supportsSampling(options, rows, width));
                        assertThrows(UnsupportedOperationException.class, () -> Ops.sample(input, options, 0));
                    } else {
                        for (long seed : new long[]{42, -1L, 0x8000000080000000L}) for (int kind = 0; kind < 3; kind++) {
                            var options = new SamplingOptions(.7f, kind == 0 ? 0 : 17, kind == 2 ? .85f : 1, seed);
                            long step = kind == 2 ? 0xffff_ffffL : 17;
                            var settings = new Sampler.Settings(options.temperature(), options.topK(), options.topP(), seed);
                            int[] expected = new int[rows];
                            for (int row = 0; row < rows; row++) { expected[row] = Sampler.reference(logits, row * width, width, settings, row, (int) step); }
                            try (var first = Ops.sample(input, options, step); var second = Ops.sample(input, options, step)) {
                                assertArrayEquals(expected, first.toIntArray()); assertArrayEquals(expected, second.toIntArray());
                            }
                        }
                        assertEquals("dynamis-gpu-primitives", backend.executedTiers().get(Operation.SAMPLE));
                        assertTrue(backend.accuracyContracts().get(Operation.SAMPLE).reference().contains("Sampler.reference"));
                    }
                }
                empty(context, backend);
            }
        }
    }

    @Test void wideVocabularyRestrictedSamplingExercisesUpstreamCandidateMergeAndFilterPaths() throws Exception {
        int width = 70001; float[] values = new float[width];
        for (int i = 0; i < width; i++) { values[i] = (i % 997) * .003f - 1; }
        var options = new SamplingOptions(.8f, 40, .95f, 1234);
        int expected = Sampler.reference(values, 0, width, new Sampler.Settings(.8f, 40, .95f, 1234), 0, 7);
        for (var mode : new GpuMode[]{GpuMode.THROUGHPUT}) {
            var backend = new GpuBackend(device(), mode);
            try (var context = ExecutionContext.create(backend)) {
                try (var logits = context.tensor(values, Shape.of(width)); var selected = Ops.sample(logits, options, 7)) { assertArrayEquals(new int[]{expected}, selected.toIntArray()); }
                empty(context, backend);
            }
        }
    }

    @Test void emptyRowsAndUnsupportedPoliciesHaveExplicitPreflightBehavior() throws Exception {
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var context = ExecutionContext.create(backend)) {
                try (var input = context.tensor(new float[0], Shape.of(0, 7)); var selected = Ops.sample(input, SamplingOptions.greedy(), 0)) { assertArrayEquals(new int[0], selected.toIntArray()); }
                try (var input = context.tensor(new float[]{1, 2, 3}, Shape.of(3))) {
                    context.synchronize(); int before = backend.liveBufferHandles();
                    assertThrows(UnsupportedOperationException.class, () -> Ops.sample(input, new SamplingOptions(1, 0, .9f, 0), 0));
                    assertThrows(UnsupportedOperationException.class, () -> Ops.sample(input, new SamplingOptions(1, 1025, 1, 0), 0));
                    assertThrows(UnsupportedOperationException.class, () -> Ops.sample(input, new SamplingOptions(Float.MIN_VALUE, 0, 1, 0), 0));
                    assertThrows(UnsupportedOperationException.class, () -> Ops.sample(input, SamplingOptions.greedy(), 0x1_0000_0000L));
                    assertEquals(before, backend.liveBufferHandles());
                    assertFalse(context.supportsSampling(SamplingOptions.greedy(), 1, Sampler.MAX_VOCAB + 1L));
                    try (var tape = context.gradientTape()) { assertThrows(IllegalStateException.class, () -> Ops.sample(input, SamplingOptions.greedy(), 0)); }
                }
                empty(context, backend);
            }
        }
    }

    @Test void nanPositiveInfinityAndAllBannedRowsFailThroughCompletion() throws Exception {
        for (var mode : GpuMode.values()) for (float[] values : new float[][]{{1, Float.NaN, 2}, {1, Float.POSITIVE_INFINITY, -1}, {Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY}}) {
            var backend = new GpuBackend(device(), mode); var context = ExecutionContext.create(backend);
            try {
                try (var input = context.tensor(values, Shape.of(3))) {
                    context.synchronize();
                    var failure = assertThrows(RuntimeException.class, () -> { try (var selected = Ops.sample(input, SamplingOptions.greedy(), 0)) { selected.toIntArray(); } });
                    assertTrue(kernelError(failure));
                }
            } finally { assertThrows(RuntimeException.class, context::close); assertEquals(0, context.liveAllocations()); assertEquals(0, backend.liveBufferHandles()); }
        }
    }

    @Test void deviceModelGenerationNeverDownloadsLogitsOrUploadsTheSampledScalarToken() throws Exception {
        for (var mode : GpuMode.values()) {
            var gpu = new GpuBackend(device(), mode); var guarded = new AtomicBoolean(); var reads = new AtomicInteger(); var uploads = new AtomicInteger();
            var wrapped = (Backend) Proxy.newProxyInstance(getClass().getClassLoader(), new Class<?>[]{Backend.class}, (proxy, method, args) -> {
                if (guarded.get() && method.getName().equals("download")) {
                    if (args[1] instanceof float[]) { throw new AssertionError("Generation downloaded a logit row"); }
                    if (args[1] instanceof int[] values) { assertEquals(1, values.length); assertEquals(1, ((KernelTensor) args[0]).buffer().elements()); reads.incrementAndGet(); }
                }
                if (guarded.get() && method.getName().equals("upload") && args[1] instanceof int[] values && values.length == 1) { uploads.incrementAndGet(); }
                try { return method.invoke(gpu, args); } catch (InvocationTargetException error) { throw error.getCause(); }
            });
            Path path = Path.of(getClass().getResource("/pretrained/tied-llama").toURI()); var checkpoint = LlamaCheckpoint.open(path);
            try (var cpu = ExecutionContext.cpu(); var context = ExecutionContext.create(wrapped)) {
                try (var control = checkpoint.load(cpu, 16); var model = checkpoint.load(context, 16)) {
                    control.training(false); model.training(false); int before = context.liveAllocations(); guarded.set(true);
                    try (var expected = GenerationSession.builder().model(control).maxNewTokens(4).sampling(s -> s.temperature(0)).build();
                         var actual = DeviceGenerationSession.builder().model(model).maxNewTokens(4).sampling(s -> s.temperature(0)).build()) {
                        int[] ids = expected.generate(new int[]{2, 5}).tokenIds();
                        assertArrayEquals(ids, actual.generate(new int[]{2, 5}).tokenIds());
                        assertArrayEquals(ids, actual.generate(new int[]{2, 5}).tokenIds());
                        assertEquals(8, reads.get()); assertEquals(0, uploads.get()); assertEquals(before, context.liveAllocations());
                    }
                }
                empty(context, gpu);
            }
        }
    }

    @Test void deviceMinimumLengthMasksStopsAndIncludesTheCommittedStopToken() throws Exception {
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var context = ExecutionContext.create(backend)) {
                var model = new ConstantModel(context);
                try (var session = DeviceGenerationSession.builder().model(model).maxNewTokens(5).minNewTokens(2).stopTokenIds(1).sampling(s -> s.temperature(0)).build()) {
                    var result = session.generate(new int[]{0}, token -> {
                        assertEquals(token.index() + 2, model.length); return GenerationSession.Decision.CONTINUE;
                    });
                    assertArrayEquals(new int[]{2, 2, 1}, result.tokenIds()); assertEquals(GenerationSession.StopReason.STOP_TOKEN, result.reason()); assertEquals(1, model.closed);
                }
                empty(context, backend);
            }
        }
    }

    @Test void stochasticModelRequestsAreRepeatableWithinEachProviderAndDevice() throws Exception {
        for (var mode : new GpuMode[]{GpuMode.THROUGHPUT}) {
            var backend = new GpuBackend(device(), mode);
            try (var context = ExecutionContext.create(backend)) {
                var model = new ConstantModel(context);
                try (var session = DeviceGenerationSession.builder().model(model).maxNewTokens(8).sampling(s -> s.temperature(.8f).topK(3).topP(.9f).seed(-1)).build()) {
                    assertEquals(session.generate(new int[]{0}).tokens(), session.generate(new int[]{0}).tokens()); assertEquals(2, model.closed);
                }
                empty(context, backend);
            }
        }
    }

    @Test void abortedSamplerRecordingsReleaseEveryScratchAllocation() throws Exception {
        for (var mode : GpuMode.values()) for (int fail = 1; fail <= 6; fail++) {
            var nativeDevice = device(); var count = new AtomicInteger(); var failAt = new AtomicInteger(Integer.MAX_VALUE);
            var wrapped = (GpuDevice) Proxy.newProxyInstance(getClass().getClassLoader(), new Class<?>[]{GpuDevice.class}, (proxy, method, args) -> {
                if (method.getName().equals("allocate") && args[0] instanceof org.dynamisengine.gpu.api.compute.BufferDescriptor descriptor
                        && descriptor.label().equals("nnjava") && count.incrementAndGet() == failAt.get()) { throw new IllegalStateException("Injected sampler scratch failure"); }
                try { return method.invoke(nativeDevice, args); } catch (InvocationTargetException error) { throw error.getCause(); }
            });
            var backend = new GpuBackend(wrapped, mode); var context = ExecutionContext.create(backend);
            try {
                try (var input = context.tensor(new float[]{1, 2, 3}, Shape.of(3))) {
                    context.synchronize(); int before = backend.liveBufferHandles(); failAt.set(count.get() + fail + 1);
                    assertThrows(IllegalStateException.class, () -> Ops.sample(input, SamplingOptions.greedy(), 0)); assertEquals(before, backend.liveBufferHandles());
                }
            } finally { assertThrows(RuntimeException.class, context::close); assertEquals(0, backend.liveBufferHandles()); }
        }
    }

    private static boolean kernelError(Throwable failure) {
        if (failure instanceof ComputeException compute && compute.code() == ComputeException.Code.KERNEL_ERROR) { return true; }
        if (failure.getCause() != null && kernelError(failure.getCause())) { return true; }
        for (var suppressed : failure.getSuppressed()) { if (kernelError(suppressed)) { return true; } } return false;
    }
    private static final class ConstantModel implements TensorAutoregressiveModel {
        private final ExecutionContext context; private int length, closed;
        ConstantModel(ExecutionContext context) { this.context = context; }
        public ExecutionContext executionContext() { return context; } public int vocabularySize() { return 3; } public int contextLength() { return 32; }
        public Cursor openSession() {
            length = 0;
            return new Cursor() {
                public int length() { return length; }
                public Tensor prefill(Tensor ids) { length = Math.toIntExact(ids.shape().elements()); return logits(); }
                public Tensor decode(Tensor id) { length++; return logits(); }
                private Tensor logits() { return context.tensor(new float[]{0, 10, 9}, Shape.of(3)); }
                public void close() { closed++; }
            };
        }
    }
}

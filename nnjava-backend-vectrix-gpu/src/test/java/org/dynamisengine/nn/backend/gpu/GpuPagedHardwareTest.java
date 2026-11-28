package org.dynamisengine.nn.backend.gpu;

import java.nio.file.Path;
import java.util.Arrays;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.dynamisengine.gpu.api.compute.BufferDescriptor;
import org.dynamisengine.gpu.api.compute.ComputeException;
import org.dynamisengine.gpu.api.compute.KernelErrors;
import org.dynamisengine.gpu.api.compute.kit.KitOptions;
import org.dynamisengine.gpu.api.compute.GpuDevice;
import org.dynamisengine.gpu.api.compute.GpuProvider;
import org.dynamisengine.nn.generation.CausalModels;
import org.dynamisengine.nn.generation.DeviceGenerationSession;
import org.dynamisengine.nn.io.StateDict;
import org.dynamisengine.nn.backend.Backend;
import org.dynamisengine.nn.nn.ParameterInitialization;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.pretrained.LlamaCheckpoint;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.transformer.CausalLanguageModel;
import org.dynamisengine.nn.transformer.InferenceOptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import static org.junit.jupiter.api.Assertions.*;

@EnabledIfSystemProperty(named = "nn.gpu.backend", matches = "METAL|VULKAN")
class GpuPagedHardwareTest {
    private static GpuDevice device() throws Exception {
        var provider = (GpuProvider) Class.forName(System.getProperty("nn.gpu.backend").equals("METAL")
                ? "org.dynamisengine.gpu.metal.MetalProvider" : "org.dynamisengine.gpu.vulkan.VulkanProvider").getConstructor().newInstance();
        assertFalse(provider.adapters().isEmpty()); return provider.adapters().getFirst().open();
    }
    private static Path fixture() throws Exception { return Path.of(GpuPagedHardwareTest.class.getResource("/pretrained/tied-llama").toURI()); }
    private static void empty(ExecutionContext context, GpuBackend backend) {
        context.synchronize(); assertEquals(0, context.liveAllocations()); assertEquals(0, backend.liveBufferHandles());
    }
    private static void closeEnough(float[] expected, float[] actual) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) { assertEquals(expected[i], actual[i], .003 + .0003 * Math.abs(expected[i]), "logit " + i); }
    }

    @Test void pagedPrefillAndRecordedGrowingDecodeMatchFp32AndIndependentTinyCheckpoint() throws Exception {
        var backend = new GpuBackend(device(), GpuMode.THROUGHPUT);
        try (var context = ExecutionContext.create(backend)) {
            try (var model = LlamaCheckpoint.open(fixture()).load(context, 16); var tokens = context.tensor(new int[]{2, 5, 3, 7}, Shape.of(1, 4))) {
                model.training(false); int baseline = context.liveAllocations();
                try (var fixed = model.newSession(1); var paged = model.newSession(1, InferenceOptions.pagedFp16());
                     var recorded = model.newSession(1, InferenceOptions.pagedFp16().withRecordedDecode());
                     var prefix = Ops.slice(tokens, 1, 0, 2); var tail = Ops.slice(tokens, 1, 2, 4)) {
                    float[] initial;
                    try (var a = fixed.prefill(prefix); var b = paged.prefill(prefix); var c = recorded.prefill(prefix)) {
                        initial = c.toFloatArray(); closeEnough(a.toFloatArray(), b.toFloatArray()); assertArrayEquals(b.toFloatArray(), initial);
                    }
                    for (int i = 0; i < 2; i++) try (var token = Ops.slice(tail, 1, i, i + 1);
                         var a = fixed.decode(token); var b = paged.decode(token); var c = recorded.decode(token)) {
                        closeEnough(a.toFloatArray(), c.toFloatArray()); assertArrayEquals(b.toFloatArray(), c.toFloatArray());
                    }
                    assertEquals(4, recorded.length()); assertEquals(1, recorded.replayStats().recordings()); assertEquals(2, recorded.replayStats().runs());
                    assertTrue(recorded.replayStats().dispatches() > 20);
                    recorded.reset(); try (var reset = recorded.prefill(prefix)) { assertArrayEquals(initial, reset.toFloatArray()); }
                }
                assertEquals(baseline, context.liveAllocations());
            }
            empty(context, backend);
        }
    }

    @Test void fullDecoderReplayRebuildsOnlyAtPageBoundariesAndPreservesReturnedLogits() throws Exception {
        var backend = new GpuBackend(device(), GpuMode.THROUGHPUT);
        var definition = CausalLanguageModel.builder().vocabularySize(19).modelWidth(8).blockCount(2).contextLength(70)
                .attention(a -> a.queryHeads(2).keyValueHeads(1)).feedForward(f -> f.hiddenWidth(12)).seed(17).define();
        try (var cpu = ExecutionContext.cpu(); var context = ExecutionContext.create(backend)) {
            try (var original = definition.build(cpu); var model = definition.build(context, ParameterInitialization.ZEROS)) {
                StateDict.capture(original).applyTo(model); model.training(false); original.training(false);
                int baseline = context.liveAllocations(); int[] prompt = new int[30]; Arrays.fill(prompt, 2);
                try (var input = context.tensor(prompt, Shape.of(1, 30)); var token = context.tensor(new int[]{3}, Shape.of(1, 1));
                     var fixed = model.newSession(1); var plain = model.newSession(1, InferenceOptions.pagedFp16());
                     var replay = model.newSession(1, InferenceOptions.pagedFp16().withRecordedDecode());
                     var a = fixed.prefill(input); var b = plain.prefill(input); var saved = replay.prefill(input)) {
                    float[] snapshot = saved.toFloatArray(); closeEnough(a.toFloatArray(), snapshot); assertArrayEquals(b.toFloatArray(), snapshot);
                    for (int length = 31; length <= 70; length++) {
                        try (var x = fixed.decode(token); var y = plain.decode(token); var z = replay.decode(token)) {
                            closeEnough(x.toFloatArray(), z.toFloatArray()); assertArrayEquals(y.toFloatArray(), z.toFloatArray(), "length " + length);
                        }
                    }
                    assertArrayEquals(snapshot, saved.toFloatArray());
                    assertEquals(3, replay.replayStats().recordings()); assertEquals(40, replay.replayStats().runs());
                    assertThrows(IllegalArgumentException.class, () -> replay.decode(token)); assertEquals(70, replay.length());
                }
                assertEquals(baseline, context.liveAllocations());
            }
            empty(context, backend);
        }
    }

    @Test void recordedDeviceGenerationMatchesPagedTokensAndCleansUpEveryRequest() throws Exception {
        var backend = new GpuBackend(device(), GpuMode.THROUGHPUT);
        try (var context = ExecutionContext.create(backend)) {
            try (var model = LlamaCheckpoint.open(fixture()).load(context, 16)) {
                model.training(false); int baseline = context.liveAllocations();
                try (var plain = DeviceGenerationSession.builder().model(CausalModels.adaptDevice(model, InferenceOptions.pagedFp16()))
                        .sampling(s -> s.temperature(0)).maxNewTokens(4).build();
                     var replay = DeviceGenerationSession.builder().model(CausalModels.adaptDevice(model, InferenceOptions.pagedFp16().withRecordedDecode()))
                        .sampling(s -> s.temperature(0)).maxNewTokens(4).build()) {
                    var expected = plain.generate(new int[]{2, 5}).tokenIds();
                    for (int i = 0; i < 2; i++) { assertArrayEquals(expected, replay.generate(new int[]{2, 5}).tokenIds()); assertEquals(baseline, context.liveAllocations()); }
                }
            }
            empty(context, backend);
        }
    }

    @Test void policyMaskBatchChunkAndStaleWeightChecksPrecedeCacheMutation() throws Exception {
        for (var mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var context = ExecutionContext.create(backend)) {
                try (var model = LlamaCheckpoint.open(fixture()).load(context, 16); var tokens = context.tensor(new int[]{2, 5}, Shape.of(1, 2));
                     var mask = context.tensor(new boolean[]{true, true}, Shape.of(1, 1, 1, 2))) {
                    model.training(false); int baseline = context.liveAllocations();
                    if (mode == GpuMode.EXACT) {
                        assertThrows(UnsupportedOperationException.class, () -> model.newSession(1, InferenceOptions.pagedFp16()));
                        assertThrows(UnsupportedOperationException.class, () -> model.newSession(1, InferenceOptions.pagedFp16().withRecordedDecode()));
                    } else {
                        assertThrows(UnsupportedOperationException.class, () -> model.newSession(2, InferenceOptions.pagedFp16()));
                        try (var session = model.newSession(1, InferenceOptions.pagedFp16().withRecordedDecode())) {
                            assertThrows(UnsupportedOperationException.class, () -> session.prefill(tokens, mask)); assertEquals(0, session.length());
                            try (var logits = session.prefill(tokens)) { logits.toFloatArray(); }
                            assertThrows(IllegalArgumentException.class, () -> session.decode(tokens)); assertEquals(2, session.length());
                            try (var token = Ops.slice(tokens, 1, 0, 1); var logits = session.decode(token); var weight = model.namedParameters().get("embedding.weight").value()) {
                                context.fill(weight, 0); assertThrows(IllegalStateException.class, () -> session.decode(token));
                            }
                            session.reset(); try (var zero = session.prefill(tokens)) { assertArrayEquals(new float[38], zero.toFloatArray()); }
                        }
                    }
                    assertEquals(baseline, context.liveAllocations());
                }
                empty(context, backend);
            }
        }
    }

    @Test void subsequentDecodeUsesTheRecordingWithoutRerunningTheJavaModel() throws Exception {
        var backend = new GpuBackend(device(), GpuMode.THROUGHPUT); var calls = new AtomicInteger();
        var monitored = (Backend) Proxy.newProxyInstance(Backend.class.getClassLoader(), new Class<?>[]{Backend.class}, (proxy, method, args) -> {
            if (method.getName().equals("execute")) { calls.incrementAndGet(); }
            try { return method.invoke(backend, args); } catch (InvocationTargetException error) { throw error.getCause(); }
        });
        try (var context = ExecutionContext.create(monitored)) {
            try (var model = LlamaCheckpoint.open(fixture()).load(context, 16); var prompt = context.tensor(new int[]{2, 5}, Shape.of(1, 2));
                 var token = context.tensor(new int[]{3}, Shape.of(1, 1))) {
                model.training(false);
                try (var session = model.newSession(1, InferenceOptions.pagedFp16().withRecordedDecode()); var prefix = session.prefill(prompt);
                     var first = session.decode(token)) {
                    float[] saved = first.toFloatArray(); int before = calls.get();
                    try (var second = session.decode(token)) { second.toFloatArray(); }
                    assertEquals(2, calls.get() - before, "Only input/output device copies should run through ordinary dispatch");
                    assertArrayEquals(saved, first.toFloatArray()); assertEquals(2, session.replayStats().runs());
                }
            }
            empty(context, backend);
        }
    }

    @Test void deferredChecksPreserveEarlierErrorsWhenAPrimitiveResetsItsReportBuffer() throws Exception {
        try (var backend = new GpuBackend(device(), GpuMode.THROUGHPUT)) {
            try (var capture = (GpuDecodeCapture) backend.beginCommandCapture();
                 var report = deviceBuffer(backend)) {
                capture.fill(report, 1); capture.checkErrors(report); capture.resetErrors(report); capture.checkErrors(report);
                try (var recorded = capture.finish()) {
                    var error = assertThrows(ComputeException.class, () -> recorded.run(0).await());
                    assertEquals(ComputeException.Code.KERNEL_ERROR, error.code());
                }
            }
        }
    }
    private static org.dynamisengine.gpu.api.compute.GpuBuffer deviceBuffer(GpuBackend backend) {
        // Obtain the same device through an owned NNJava buffer, with independent native lifetime.
        var buffer = backend.allocate(org.dynamisengine.nn.tensor.DType.INT32, 4);
        try { return backend.storage(new org.dynamisengine.nn.backend.KernelTensor(buffer,
                org.dynamisengine.nn.tensor.TensorLayout.contiguous(Shape.of(4)))).slice(0, KernelErrors.BYTES); }
        finally { buffer.close(); }
    }

    @Test void recordedInvalidDeviceTokenFailsBeforeLengthCommitAndReleasesThePlan() throws Exception {
        var backend = new GpuBackend(device(), GpuMode.THROUGHPUT);
        var context = ExecutionContext.create(backend);
        try {
            try (var model = LlamaCheckpoint.open(fixture()).load(context, 16); var prompt = context.tensor(new int[]{2, 5}, Shape.of(1, 2));
                 var valid = context.tensor(new int[]{3}, Shape.of(1, 1)); var invalid = context.tensor(new int[]{19}, Shape.of(1, 1))) {
                model.training(false);
                try (var session = model.newSession(1, InferenceOptions.pagedFp16().withRecordedDecode());
                     var prefix = session.prefill(prompt); var first = session.decode(valid)) {
                    assertThrows(ComputeException.class, () -> session.decode(invalid));
                    assertThrows(IllegalStateException.class, session::length);
                }
            }
            assertEquals(0, context.liveAllocations()); assertEquals(0, backend.liveBufferHandles());
        } finally { assertThrows(IllegalStateException.class, context::close); }
    }

    @Test void failedRecordingAllocationsAbortEveryIntermediateAndReleaseCacheStorage() throws Exception {
        for (String label : new String[]{"nnjava-replay-position", "nnjava-replay-error", "nnjava"})
            for (int failAt : label.equals("nnjava-replay-error") ? new int[]{1} : new int[]{1, 2}) {
            var gpu = device(); var armed = new AtomicBoolean(); var allocations = new AtomicInteger();
            var monitored = (GpuDevice) Proxy.newProxyInstance(GpuDevice.class.getClassLoader(), new Class<?>[]{GpuDevice.class}, (proxy, method, args) -> {
                if (armed.get() && method.getName().equals("allocate") && args[0] instanceof BufferDescriptor descriptor
                        && descriptor.label().equals(label) && allocations.incrementAndGet() == failAt) { throw new IllegalStateException("Injected recording failure"); }
                try { return method.invoke(gpu, args); } catch (InvocationTargetException error) { throw error.getCause(); }
            });
            var backend = new GpuBackend(monitored, GpuMode.THROUGHPUT); var context = ExecutionContext.create(backend);
            try {
                try (var model = LlamaCheckpoint.open(fixture()).load(context, 16); var prompt = context.tensor(new int[]{2, 5}, Shape.of(1, 2));
                     var token = context.tensor(new int[]{3}, Shape.of(1, 1))) {
                    model.training(false);
                    try (var session = model.newSession(1, InferenceOptions.pagedFp16().withRecordedDecode()); var prefix = session.prefill(prompt)) {
                        armed.set(true); assertThrows(IllegalStateException.class, () -> session.decode(token));
                    }
                }
                assertEquals(0, context.liveAllocations()); assertEquals(0, backend.liveBufferHandles());
            } finally {
                try { context.close(); } catch (IllegalStateException expected) { assertTrue(expected.getMessage().contains("Backend execution failed")); }
            }
        }
    }

    @Test void debugKernelsAndChunkedPagedPrefillRetainTheSameCausalOffset() throws Exception {
        var gpu = device();
        var fixture = fixture();
        new KitOptions(false, true).run(() -> {
            var backend = new GpuBackend(gpu, GpuMode.THROUGHPUT);
            try (var context = ExecutionContext.create(backend)) {
                try (var model = LlamaCheckpoint.open(fixture).load(context, 16); var tokens = context.tensor(new int[]{2, 5, 3, 7}, Shape.of(1, 4))) {
                    model.training(false);
                    try (var plain = model.newSession(1, InferenceOptions.pagedFp16());
                         var replay = model.newSession(1, InferenceOptions.pagedFp16().withRecordedDecode());
                         var first = Ops.slice(tokens, 1, 0, 1); var rest = Ops.slice(tokens, 1, 1, 4);
                         var full = plain.prefill(tokens); var prefix = replay.prefill(first)) {
                        for (int i = 0; i < 3; i++) try (var token = Ops.slice(rest, 1, i, i + 1); var next = replay.decode(token);
                             var expected = Ops.slice(full, 1, i + 1, i + 2)) { closeEnough(expected.toFloatArray(), next.toFloatArray()); }
                    }
                    try (var session = model.newSession(1, InferenceOptions.pagedFp16()); var first = Ops.slice(tokens, 1, 0, 2);
                         var rest = Ops.slice(tokens, 1, 2, 4); var prefix = session.prefill(first); var next = session.decode(rest); var full = model.forward(tokens)) {
                        try (var expected = Ops.slice(full, 1, 2, 4)) { closeEnough(expected.toFloatArray(), next.toFloatArray()); }
                    }
                } catch (java.io.IOException error) { throw new RuntimeException(error); }
                empty(context, backend);
            }
        });
    }

    @Test void contextCloseOwnsOutstandingNativeCacheAndRecordingResources() throws Exception {
        var backend = new GpuBackend(device(), GpuMode.THROUGHPUT); var context = ExecutionContext.create(backend);
        var model = LlamaCheckpoint.open(fixture()).load(context, 16); model.training(false);
        var session = model.newSession(1, InferenceOptions.pagedFp16().withRecordedDecode());
        try {
            try (var prompt = context.tensor(new int[]{2, 5}, Shape.of(1, 2)); var token = context.tensor(new int[]{3}, Shape.of(1, 1));
                 var prefix = session.prefill(prompt); var next = session.decode(token)) {
                assertEquals(1, session.replayStats().runs());
            }
            context.close(); assertEquals(0, backend.liveBufferHandles());
        } finally { session.close(); model.close(); context.close(); }
    }
}

package org.dynamisengine.nn.generation;

import java.util.ArrayList;
import java.util.List;
import org.dynamisengine.nn.backend.Backend;
import org.dynamisengine.nn.backend.Buffer;
import org.dynamisengine.nn.backend.Completion;
import org.dynamisengine.nn.backend.KernelCall;
import org.dynamisengine.nn.backend.KernelTensor;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.ops.SamplingOptions;
import org.dynamisengine.nn.runtime.Device;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;
import org.dynamisengine.nn.tensor.TensorAccess;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/** Scripted tensor provider tests the generation protocol, with no floating-point download implementation. */
class DeviceGenerationSessionTest {
    @Test void tokensCommitBeforeCallbacksAndFeedTheSameProviderBufferIntoDecode() {
        var backend = new Scripted();
        try (var context = ExecutionContext.create(backend)) {
            var model = new Model(context, backend);
            try (var session = greedy(model, 4).stopTokenIds(1).build()) {
                var result = session.generate(new int[]{0}, token -> {
                    assertEquals(token.index() + 2, model.length); return GenerationSession.Decision.CONTINUE;
                });
                assertArrayEquals(new int[]{2, 1}, result.tokenIds()); assertArrayEquals(new int[]{0, 2, 1}, result.allTokenIds());
                assertEquals(GenerationSession.StopReason.STOP_TOKEN, result.reason()); assertEquals(2, backend.tokenReads);
                assertEquals(1, backend.intUploads); assertEquals(1, model.closed); assertEquals(0, context.liveAllocations());
            }
        }
    }

    @Test void freshRequestsResetTheProviderStepAndOwnIndependentCursors() {
        var backend = new Scripted();
        try (var context = ExecutionContext.create(backend)) {
            var model = new Model(context, backend);
            try (var session = DeviceGenerationSession.builder().model(model).maxNewTokens(3).sampling(s -> s.seed(-1)).build()) {
                assertEquals(session.generate(new int[]{0}).tokens(), session.generate(new int[]{0}).tokens());
                assertEquals(List.of(0L, 1L, 2L, 0L, 1L, 2L), backend.steps); assertEquals(List.of(-1L, -1L, -1L, -1L, -1L, -1L), backend.seeds);
                assertEquals(2, model.opened); assertEquals(2, model.closed); assertEquals(0, context.liveAllocations());
            }
        }
    }

    @Test void sequenceStopsAndCriteriaRunOnCommittedDeviceTokens() {
        var backend = new Scripted();
        try (var context = ExecutionContext.create(backend)) {
            var model = new Model(context, backend);
            try (var session = greedy(model, 5).stopSequence(1, 2).build()) {
                var result = session.generate(new int[]{1}); assertArrayEquals(new int[]{2, 1, 2}, result.tokenIds());
                assertEquals(GenerationSession.StopReason.STOP_SEQUENCE, result.reason());
            }
            try (var session = greedy(model, 5).stopping(history -> history.generatedCount() == 2).build()) {
                assertEquals(GenerationSession.StopReason.CRITERION, session.generate(new int[]{0}).reason());
            }
            assertEquals(0, context.liveAllocations());
        }
    }

    @Test void callbackStopCancelAndReentryRulesPreserveCursorOwnership() throws Exception {
        var backend = new Scripted();
        try (var context = ExecutionContext.create(backend)) {
            var model = new Model(context, backend);
            try (var session = greedy(model, 4).build(); var executor = java.util.concurrent.Executors.newSingleThreadExecutor()) {
                assertEquals(GenerationSession.StopReason.CALLBACK, session.generate(new int[]{0}, token -> {
                    assertThrows(IllegalStateException.class, () -> session.generate(new int[]{0})); assertThrows(IllegalStateException.class, session::close);
                    return GenerationSession.Decision.STOP;
                }).reason());
                executor.submit(session::cancel).get(); assertEquals(GenerationSession.StopReason.CANCELLED, session.generate(new int[]{0}).reason());
                assertEquals(1, model.opened);
                assertEquals(GenerationSession.StopReason.CANCELLED, session.generate(new int[]{0}, token -> { session.cancel(); return GenerationSession.Decision.CONTINUE; }).reason());
                executor.submit(() -> assertThrows(IllegalStateException.class, () -> session.generate(new int[]{0}))).get();
                assertEquals(2, model.closed); assertEquals(0, context.liveAllocations());
            }
        }
    }

    @Test void failedCallbacksAndDecodeCloseEveryTensorAndAllowAFreshRequest() {
        var backend = new Scripted();
        try (var context = ExecutionContext.create(backend)) {
            var model = new Model(context, backend);
            try (var session = greedy(model, 2).build()) {
                assertThrows(IllegalArgumentException.class, () -> session.generate(new int[]{0}, token -> { throw new IllegalArgumentException("callback"); }));
                assertEquals(0, context.liveAllocations()); model.failDecode = true;
                assertThrows(IllegalArgumentException.class, () -> session.generate(new int[]{0})); assertEquals(0, context.liveAllocations());
                model.failDecode = false; assertEquals(2, session.generate(new int[]{0}).tokens().size()); assertEquals(3, model.closed);
                assertEquals(0, context.liveAllocations());
            }
        }
    }

    @Test void invalidMetadataPoliciesPromptsAndEmptyBudgetsPreflightBeforeOpeningCaches() {
        var backend = new Scripted();
        try (var context = ExecutionContext.create(backend)) {
            var model = new Model(context, backend);
            assertThrows(UnsupportedOperationException.class, () -> greedy(model, 2).sampling(s -> s.repetitionPenalty(2)).build());
            assertThrows(UnsupportedOperationException.class, () -> greedy(model, 2).minNewTokens(1).stopTokenIds(1).build());
            try (var session = greedy(model, 2).build()) {
                assertThrows(IllegalArgumentException.class, () -> session.generate(new int[0]));
                assertThrows(IllegalArgumentException.class, () -> session.generate(new int[]{3}));
                assertThrows(IllegalArgumentException.class, () -> session.generate(new int[16]));
                try (var tape = context.gradientTape()) { assertThrows(IllegalStateException.class, () -> session.generate(new int[]{0})); }
            }
            try (var session = greedy(model, 0).build()) { assertEquals(0, session.generate(new int[]{0}).tokens().size()); }
            assertEquals(0, model.opened); assertEquals(0, context.liveAllocations());
        }
    }

    @Test void malformedCursorLogitsReleaseTheUnreturnedViewAndCache() {
        var backend = new Scripted();
        try (var context = ExecutionContext.create(backend)) {
            var model = new Model(context, backend); model.badWidth = true;
            try (var session = greedy(model, 2).build()) {
                assertThrows(IllegalArgumentException.class, () -> session.generate(new int[]{0}));
                assertEquals(1, model.closed); assertEquals(0, context.liveAllocations());
            }
        }
    }

    private static DeviceGenerationSession.Builder greedy(Model model, int maximum) {
        return DeviceGenerationSession.builder().model(model).maxNewTokens(maximum).sampling(s -> s.temperature(0));
    }
    private static final class Model implements TensorAutoregressiveModel {
        private final ExecutionContext context; private final Scripted backend;
        int opened, closed, length; boolean failDecode, badWidth;
        Model(ExecutionContext context, Scripted backend) { this.context = context; this.backend = backend; }
        public ExecutionContext executionContext() { return context; }
        public int vocabularySize() { return 3; }
        public int contextLength() { return 16; }
        public Cursor openSession() {
            opened++; length = 0;
            return new Cursor() {
                public int length() { return length; }
                public Tensor prefill(Tensor input) { length = Math.toIntExact(input.shape().elements()); return logits(); }
                public Tensor decode(Tensor token) {
                    assertSame(backend.selected, TensorAccess.kernel(token).buffer(), "Sampled storage must directly feed decode");
                    if (failDecode) { throw new IllegalArgumentException("decode"); } length++; return logits();
                }
                private Tensor logits() { return context.tensor(new float[badWidth ? 2 : 3], Shape.of(badWidth ? 2 : 3)); }
                public void close() { closed++; }
            };
        }
    }
    private static final class Scripted implements Backend {
        private final Device device = new Device("scripted", "test");
        private final List<Long> steps = new ArrayList<>(), seeds = new ArrayList<>();
        private Storage selected; private int tokenReads, intUploads;
        public Device device() { return device; }
        public boolean supports(Operation operation, DType dtype) { return operation == Operation.SAMPLE && dtype == DType.INT32; }
        public boolean supportsSampling(SamplingOptions options, long rows, long vocabulary) { return rows == 1 && vocabulary == 3; }
        public Buffer allocate(DType type, long elements) { return new Storage(type, Math.toIntExact(elements)); }
        public Completion execute(KernelCall call) {
            assertEquals(Operation.SAMPLE, call.operation()); long step = call.attributes().integer("step"); steps.add(step); seeds.add(call.attributes().integer("seed"));
            selected = (Storage) call.output().buffer(); ((int[]) selected.values)[0] = step % 2 == 0 ? 2 : 1; return Completion.completed();
        }
        public Completion upload(Buffer target, float[] source) { System.arraycopy(source, 0, ((Storage) target).values, 0, source.length); return Completion.completed(); }
        public Completion upload(Buffer target, int[] source) { intUploads++; System.arraycopy(source, 0, ((Storage) target).values, 0, source.length); return Completion.completed(); }
        public Completion upload(Buffer target, boolean[] source) { throw new AssertionError("No masks in this protocol provider"); }
        public Completion download(KernelTensor source, float[] target) { throw new AssertionError("Device generation must never download logits"); }
        public Completion download(KernelTensor source, int[] target) {
            tokenReads++; assertEquals(1, target.length); target[0] = ((int[]) ((Storage) source.buffer()).values)[Math.toIntExact(source.layout().offset())]; return Completion.completed();
        }
        public Completion download(KernelTensor source, boolean[] target) { throw new AssertionError("Unexpected mask download"); }
        public Completion randomNormal(Buffer target, long seed, long offset, float deviation) { throw new AssertionError("No initialization in this provider"); }
        public void close() { }
        private final class Storage implements Buffer {
            private final DType type; private final int count; private final Object values; private boolean closed;
            Storage(DType type, int count) { this.type = type; this.count = count; values = type == DType.FP32 ? new float[count] : new int[count]; }
            public Device device() { return device; } public DType dtype() { return type; } public long elements() { return count; }
            public boolean isClosed() { return closed; } public void close() { closed = true; }
        }
    }
}

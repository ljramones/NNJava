package org.dynamisengine.nn;

import org.dynamisengine.nn.backend.Buffer;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;

class ExecutionContextTest {
    @Test
    void pendingCommandsRetainClosedInputsAndSnapshotHostUploads() {
        var backend = new DeferredBackend();
        try (var context = ExecutionContext.create(backend)) {
            float[] source = {1, 2};
            Tensor a = context.tensor(source, Shape.of(2));
            Tensor b = context.tensor(new float[]{3, 4}, Shape.of(2));
            source[0] = 100;
            Tensor result = Ops.add(a, b);
            a.close();
            b.close();
            assertEquals(3, context.liveAllocations());
            assertTrue(backend.allocations.stream().noneMatch(Buffer::isClosed));
            assertEquals(Operation.ADD, backend.calls.getFirst().operation());
            assertArrayEquals(new float[]{4, 6}, result.toFloatArray());
            assertEquals(1, context.liveAllocations());
            result.close();
            assertEquals(0, context.liveAllocations());
            assertTrue(backend.allocations.stream().allMatch(Buffer::isClosed));
        }
        assertTrue(backend.closed);
    }

    @Test
    void backwardWorksWithDeferredExecutionAndClosedIntermediateHandles() {
        try (var context = ExecutionContext.create(new DeferredBackend());
             var x = context.tensor(new float[]{2, 3}, Shape.of(2), true);
             var survivor = x.retain(); var tape = context.gradientTape()) {
            Tensor squared = Ops.square(x);
            try (Tensor total = Ops.sum(squared)) {
                squared.close();
                x.close();
                tape.backward(total);
            }
            try (Tensor gradient = survivor.gradient()) {
                assertArrayEquals(new float[]{4, 6}, gradient.toFloatArray());
            }
            survivor.clearGradient();
            assertEquals(1, context.liveAllocations());
        }
    }

    @Test
    void deferredFailurePoisonsContextButStillReleasesAllWorkAndAllocations() {
        var backend = new DeferredBackend();
        var context = ExecutionContext.create(backend);
        Tensor x = context.tensor(new float[]{2}, Shape.of());
        backend.failNextKernel = true;
        Tensor result = Ops.square(x);
        x.close();
        result.close();
        assertThrows(IllegalStateException.class, context::synchronize);
        assertEquals(0, context.liveAllocations());
        assertTrue(backend.allocations.stream().allMatch(Buffer::isClosed));
        assertThrows(IllegalStateException.class, () -> context.full(Shape.of(), 1));
        assertThrows(IllegalStateException.class, context::close);
        assertTrue(backend.closed);
        context.close();
    }

    @Test
    void closingContextDrainsQueuedWorkEvenAfterAllCallerHandlesClose() {
        var backend = new DeferredBackend();
        var context = ExecutionContext.create(backend);
        Tensor x = context.tensor(new float[]{2}, Shape.of());
        Tensor result = Ops.square(x);
        x.close();
        result.close();
        context.close();
        assertEquals(0, context.liveAllocations());
        assertTrue(backend.closed);
        assertTrue(backend.allocations.stream().allMatch(Buffer::isClosed));
    }

    @Test
    void allHostTypesSupportDeferredTransfers() {
        try (var context = ExecutionContext.create(new DeferredBackend());
             var integers = context.tensor(new int[]{1, 2}, Shape.of(2));
             var booleans = context.tensor(new boolean[]{true, false}, Shape.of(2));
             var copied = Ops.copy(booleans)) {
            assertArrayEquals(new int[]{1, 2}, integers.toIntArray());
            assertArrayEquals(new boolean[]{true, false}, copied.toBooleanArray());
        }
    }

    @Test
    void contextsRejectAccessFromOtherThreads() throws InterruptedException {
        try (var context = ExecutionContext.cpu()) {
            AtomicReference<Throwable> failure = new AtomicReference<>();
            Thread thread = Thread.ofPlatform().start(() -> {
                try {
                    context.device();
                } catch (Throwable caught) {
                    failure.set(caught);
                }
            });
            thread.join();
            assertInstanceOf(IllegalStateException.class, failure.get());
        }
    }
}

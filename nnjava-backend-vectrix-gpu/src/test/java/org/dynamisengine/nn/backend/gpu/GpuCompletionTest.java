package org.dynamisengine.nn.backend.gpu;

import static org.junit.jupiter.api.Assertions.*;
import java.time.Duration;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import org.dynamisengine.gpu.api.compute.ComputeException;
import org.dynamisengine.gpu.api.compute.GpuSubmission;
import org.junit.jupiter.api.Test;

class GpuCompletionTest {
    @Test void failedSubmissionIsClosedOnceAndNeverPublishesReadback() {
        var failure = new IllegalStateException("device execution failed");
        AtomicInteger closes = new AtomicInteger();
        GpuSubmission submission = new GpuSubmission() {
            @Override public Status status() { return Status.FAILED; }
            @Override public Optional<ComputeException> failure() { return Optional.empty(); }
            @Override public boolean await(Duration timeout) { throw failure; }
            @Override public void close() { closes.incrementAndGet(); }
        };
        var completion = new GpuCompletion(submission, () -> fail("Failed work must not publish readback"));
        assertTrue(completion.isDone());
        assertSame(failure, assertThrows(IllegalStateException.class, completion::await));
        assertSame(failure, assertThrows(IllegalStateException.class, completion::await));
        assertEquals(1, closes.get());
    }

    @Test void pendingTimeoutRetainsSubmissionAndPublishesExactlyOnceAfterCompletion() {
        AtomicInteger published = new AtomicInteger();
        class Submission implements GpuSubmission {
            int waits; boolean released;
            @Override public Status status() { return waits < 2 ? Status.PENDING : Status.COMPLETE; }
            @Override public Optional<ComputeException> failure() { return Optional.empty(); }
            @Override public boolean await(Duration timeout) {
                assertFalse(released);
                assertEquals(0, published.get());
                return ++waits >= 2;
            }
            @Override public void close() { assertEquals(Status.COMPLETE, status()); released = true; }
        }
        var submission = new Submission();
        var completion = new GpuCompletion(submission, published::incrementAndGet);
        assertFalse(completion.isDone());
        completion.await(); completion.await();
        assertTrue(completion.isDone());
        assertTrue(submission.released);
        assertEquals(2, submission.waits);
        assertEquals(1, published.get());
    }
}

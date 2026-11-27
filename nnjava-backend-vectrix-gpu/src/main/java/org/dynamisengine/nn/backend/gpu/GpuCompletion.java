package org.dynamisengine.nn.backend.gpu;

import java.time.Duration;
import org.dynamisengine.gpu.api.compute.GpuSubmission;
import org.dynamisengine.nn.backend.Completion;

/** A bounded runtime timeout is pending work, never permission to release storage. */
final class GpuCompletion implements Completion {
    private final GpuSubmission submission;
    private final Runnable publish;
    private boolean finished;
    private RuntimeException failure;

    GpuCompletion(GpuSubmission submission, Runnable publish) {
        this.submission = submission;
        this.publish = publish;
    }

    @Override public boolean isDone() {
        return finished || submission.status() != GpuSubmission.Status.PENDING;
    }

    @Override public void await() {
        if (!finished) {
            try {
                while (!submission.await(Duration.ofSeconds(30))) {
                    // Preserve the submission and all retained buffers across timeouts.
                }
                publish.run();
            } catch (RuntimeException error) {
                failure = error;
            } finally {
                try { submission.close(); }
                catch (RuntimeException error) {
                    if (failure == null) { failure = error; }
                    else if (failure != error) { failure.addSuppressed(error); }
                } finally { finished = true; }
            }
        }
        if (failure != null) { throw failure; }
    }
}

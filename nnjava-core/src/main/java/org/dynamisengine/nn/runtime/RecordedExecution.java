package org.dynamisengine.nn.runtime;

import java.util.List;
import org.dynamisengine.nn.backend.CommandRecording;
import org.dynamisengine.nn.tensor.Tensor;
import org.dynamisengine.nn.tensor.TensorAccess;

/** Owned fixed inference graph. output() aliases storage overwritten by the next run. */
public final class RecordedExecution implements AutoCloseable {
    private final ExecutionContext context;
    private final CommandRecording recording;
    private final Tensor output;
    private boolean closed, executed;
    RecordedExecution(ExecutionContext context, CommandRecording recording, Tensor output) {
        this.context = context; this.recording = recording; this.output = output;
    }
    public int dispatches() { ensureOpen(); return recording.dispatches(); }
    public void run(int position) {
        ensureOpen();
        if (position < 0) { throw new IllegalArgumentException("Negative decoder position"); }
        if (context.activeTape() != null) { throw new IllegalStateException("Recorded execution is inference-only"); }
        TensorAccess.beforeWrite(output);
        context.enqueueExternal(() -> recording.run(position), List.of(output));
        executed = true;
    }
    public Tensor output() {
        ensureOpen();
        if (!executed) { throw new IllegalStateException("Run the recording before reading its output"); }
        return output.retain();
    }
    private void ensureOpen() { context.ensureOpen(); if (closed) { throw new IllegalStateException("Recording is closed"); } }
    @Override public void close() {
        context.checkThread();
        if (!closed) {
            closed = true;
            try { recording.close(); } finally { output.close(); context.releaseResource(this); }
        }
    }
}

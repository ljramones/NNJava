package org.dynamisengine.nn.backend;

/** Fixed bindings and shapes; position is the sole varying decoder parameter. */
public interface CommandRecording extends AutoCloseable {
    int dispatches();
    Completion run(int position);
    @Override void close();
}

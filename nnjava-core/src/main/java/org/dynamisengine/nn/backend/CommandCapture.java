package org.dynamisengine.nn.backend;

/** An inference command capture; closing an unfinished capture aborts it. */
public interface CommandCapture extends AutoCloseable {
    CommandRecording finish();
    @Override void close();
}

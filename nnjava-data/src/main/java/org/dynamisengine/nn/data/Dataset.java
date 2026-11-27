package org.dynamisengine.nn.data;

import org.dynamisengine.nn.runtime.ExecutionContext;

/**
 * Applications implement this interface for file-backed or domain-specific samples.
 */
public interface Dataset extends AutoCloseable {
    long size();

    Batch batch(ExecutionContext context, int[] indices);

    @Override
    default void close() {
    }
}

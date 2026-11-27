package org.dynamisengine.nn.backend;

import org.dynamisengine.nn.runtime.Device;
import org.dynamisengine.nn.tensor.DType;

/**
 * Opaque allocation. The runtime retains it through tensor and submission ownership.
 */
public interface Buffer extends AutoCloseable {
    Device device();

    DType dtype();

    long elements();

    boolean isClosed();

    /**
     * Release the allocation; repeated calls must be harmless.
     */
    @Override
    void close();
}

package org.dynamisengine.nn.formats;

import java.lang.foreign.MemorySegment;
import java.util.Map;

/**
 * Read-only mapped tensors. Borrowed data segments become invalid when the file is closed.
 */
public interface TensorFile extends AutoCloseable {
    Map<String, Object> metadata();

    Map<String, TensorDescriptor> tensors();

    MemorySegment data(String name);

    @Override
    void close();
}

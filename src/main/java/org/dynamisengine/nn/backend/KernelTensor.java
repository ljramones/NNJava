package org.dynamisengine.nn.backend;

import java.util.Objects;
import org.dynamisengine.nn.tensor.TensorLayout;

/** A validated layout over an allocation; offsets and strides count elements. */
public record KernelTensor(Buffer buffer, TensorLayout layout) {
    public KernelTensor {
        Objects.requireNonNull(buffer, "buffer");
        Objects.requireNonNull(layout, "layout");
        if (buffer.isClosed() || layout.requiredElements() > buffer.elements()) {
            throw new IllegalArgumentException("Layout exceeds live buffer storage");
        }
    }
}

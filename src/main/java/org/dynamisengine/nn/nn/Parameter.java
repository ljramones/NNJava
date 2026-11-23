package org.dynamisengine.nn.nn;

import java.util.Objects;
import org.dynamisengine.nn.tensor.Tensor;

/** Named ownership of a trainable leaf. value() returns an owned retained handle. */
public final class Parameter implements AutoCloseable {
    private final String name;
    private final Tensor value;
    private boolean closed;

    public Parameter(String name, Tensor value) {
        this.name = Objects.requireNonNull(name, "name");
        if (name.isBlank() || !value.isLeaf()) {
            throw new IllegalArgumentException("A parameter needs a name and a trainable leaf tensor");
        }
        this.value = value.retain();
    }

    public String name() { return name; }

    public Tensor value() {
        if (closed) {
            throw new IllegalStateException("Parameter is closed");
        }
        return value.retain();
    }

    public void clearGradient() {
        try (Tensor tensor = value()) {
            tensor.clearGradient();
        }
    }

    @Override
    public void close() {
        if (!closed) {
            value.close();
            closed = true;
        }
    }
}

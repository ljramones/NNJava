package org.dynamisengine.nn.data;

import org.dynamisengine.nn.tensor.Tensor;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

/**
 * Owns the supplied handles. get() gives the caller a retained owned handle.
 */
public final class Batch implements AutoCloseable {
    private final Map<String, Tensor> fields;
    private boolean closed;

    public Batch(Map<String, Tensor> fields) {
        this.fields = Collections.unmodifiableMap(new LinkedHashMap<>(fields));
    }

    public Set<String> names() {
        ensureOpen();
        return fields.keySet();
    }

    public Tensor get(String name) {
        ensureOpen();
        Tensor value = fields.get(name);
        if (value == null) {
            throw new IllegalArgumentException("Unknown batch field " + name);
        }
        return value.retain();
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Batch is closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            fields.values().forEach(Tensor::close);
            closed = true;
        }
    }
}

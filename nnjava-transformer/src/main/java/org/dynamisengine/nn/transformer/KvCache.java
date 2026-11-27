package org.dynamisengine.nn.transformer;

import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.*;

/**
 * Fixed-capacity session state. Returned read-only views remain valid until the next cache mutation.
 */
public final class KvCache implements AutoCloseable {
    private final ExecutionContext context;
    private final Tensor keys, values;
    private final long batch, heads, capacity, features;
    private long length;
    private boolean closed;

    public KvCache(ExecutionContext context, long batch, long heads, long capacity, long features) {
        if (batch <= 0 || heads <= 0 || capacity <= 0 || features <= 0) {
            throw new IllegalArgumentException("Positive cache dimensions required");
        }
        this.context = context;
        this.batch = batch;
        this.heads = heads;
        this.capacity = capacity;
        this.features = features;
        keys = context.zeros(DType.FP32, Shape.of(batch, heads, capacity, features));
        try {
            values = context.zeros(DType.FP32, keys.shape());
        } catch (RuntimeException error) {
            keys.close();
            throw error;
        }
    }

    public long length() {
        ensureOpen();
        return length;
    }

    public long capacity() {
        ensureOpen();
        return capacity;
    }

    public void append(Tensor key, Tensor value) {
        ensureMutable();
        context.requireContext(key);
        context.requireContext(value);
        if (key.dtype() != DType.FP32 || value.dtype() != DType.FP32 || key.shape().rank() != 4 || !key.shape().equals(value.shape())
                || key.shape().dimension(0) != batch || key.shape().dimension(1) != heads || key.shape().dimension(3) != features) {
            throw new IllegalArgumentException("Cache append dimensions mismatch");
        }
        long next = Math.addExact(length, key.shape().dimension(2));
        if (next > capacity) {
            throw new IllegalArgumentException("KV cache capacity exceeded");
        }
        try (Tensor k = Ops.slice(keys, 2, length, next); Tensor v = Ops.slice(values, 2, length, next)) {
            context.assign(k, key);
            context.assign(v, value);
            context.synchronize();
            length = next;
        }
    }

    public Tensor keys() {
        ensureOpen();
        return view(keys);
    }

    public Tensor values() {
        ensureOpen();
        return view(values);
    }

    private Tensor view(Tensor source) {
        return TensorAccess.view(source, source.layout().slice(2, 0, length, 1), true);
    }

    public void reset() {
        ensureMutable();
        length = 0;
    }

    private void ensureMutable() {
        ensureOpen();
        if (context.activeTape() != null) {
            throw new IllegalStateException("KV cache mutation is inference-only");
        }
    }

    private void ensureOpen() {
        context.ensureOpen();
        if (closed) {
            throw new IllegalStateException("KV cache is closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            keys.close();
            values.close();
            closed = true;
        }
    }
}

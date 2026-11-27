package org.dynamisengine.nn.tensor;

import org.dynamisengine.nn.backend.Buffer;
import org.dynamisengine.nn.runtime.Device;
import org.dynamisengine.nn.runtime.ExecutionContext;

import java.util.concurrent.atomic.AtomicLong;

/**
 * An owned tensor handle. retain() creates another owned handle; closing a handle
 * leaves views and saved gradient values alive. Host-array access always copies.
 */
public final class Tensor implements AutoCloseable {
    final State state;
    private boolean closed;

    Tensor(State state) {
        this.state = state;
        state.references++;
    }

    public Shape shape() {
        checkOpen();
        return state.layout.shape();
    }

    public TensorLayout layout() {
        checkOpen();
        return state.layout;
    }

    public DType dtype() {
        checkOpen();
        return state.storage.buffer.dtype();
    }

    public Device device() {
        checkOpen();
        return state.context.device();
    }

    public ExecutionContext context() {
        checkOpen();
        return state.context;
    }

    public long id() {
        checkOpen();
        return state.id;
    }

    public boolean requiresGrad() {
        checkOpen();
        return state.requiresGrad;
    }

    public boolean isLeaf() {
        checkOpen();
        return state.requiresGrad && state.tapeId == 0;
    }

    public Tensor retain() {
        checkOpen();
        return new Tensor(state);
    }

    /**
     * A shared-storage value with no gradient history. Mutation rules still apply.
     */
    public Tensor detach() {
        checkOpen();
        return new Tensor(new State(state.context, state.storage, state.layout, false, state.readOnly));
    }

    public float[] toFloatArray() {
        checkOpen();
        return state.context.readFloats(this);
    }

    public int[] toIntArray() {
        checkOpen();
        return state.context.readInts(this);
    }

    public boolean[] toBooleanArray() {
        checkOpen();
        return state.context.readBooleans(this);
    }

    public short[] toShortArray() {
        checkOpen();
        return state.context.readShorts(this);
    }

    public float item() {
        if (shape().rank() != 0) {
            throw new IllegalStateException("item() requires a scalar tensor");
        }
        return toFloatArray()[0];
    }

    public boolean hasGradient() {
        checkOpen();
        return state.gradient != null;
    }

    /**
     * Return an owned handle to the accumulated leaf gradient.
     */
    public Tensor gradient() {
        checkOpen();
        if (state.gradient == null) {
            throw new IllegalStateException("Tensor has no accumulated gradient");
        }
        return state.gradient.retain();
    }

    public void clearGradient() {
        checkOpen();
        if (state.gradient != null) {
            state.gradient.close();
            state.gradient = null;
        }
    }

    void checkOpen() {
        if (closed) {
            throw new IllegalStateException("Tensor handle is closed");
        }
        state.context.ensureOpen();
    }

    @Override
    public void close() {
        state.context.checkThread();
        if (!closed) {
            closed = true;
            if (--state.references == 0) {
                if (state.gradient != null) {
                    state.gradient.close();
                    state.gradient = null;
                }
                if (--state.storage.references == 0) {
                    state.context.releaseBuffer(state.storage.buffer);
                }
            }
        }
    }

    static final class State {
        private static final AtomicLong IDS = new AtomicLong();
        final long id = IDS.incrementAndGet();
        final ExecutionContext context;
        final Storage storage;
        final TensorLayout layout;
        final boolean readOnly;
        int references;
        boolean requiresGrad;
        long tapeId;
        Tensor gradient;

        State(ExecutionContext context, Storage storage, TensorLayout layout, boolean requiresGrad, boolean readOnly) {
            this.context = context;
            this.storage = storage;
            this.layout = layout;
            this.requiresGrad = requiresGrad;
            this.readOnly = readOnly;
            storage.references++;
        }
    }

    static final class Storage {
        final Buffer buffer;
        int references;
        int savedValues;
        long version;

        Storage(Buffer buffer) {
            this.buffer = buffer;
        }
    }
}

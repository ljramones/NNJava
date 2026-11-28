package org.dynamisengine.nn.runtime;

import java.util.List;
import org.dynamisengine.nn.backend.PagedAttentionState;
import org.dynamisengine.nn.tensor.*;

/** Owned FP16 provider cache, with FP32 query/projection boundaries and explicit step commits. */
public final class PagedAttentionCache implements AutoCloseable {
    private final ExecutionContext context;
    private final PagedAttentionState.Config config;
    private final PagedAttentionState state;
    private int length, prepared;
    private boolean closed;

    PagedAttentionCache(ExecutionContext context, PagedAttentionState.Config config, PagedAttentionState state) {
        this.context = context; this.config = config; this.state = state;
    }
    public long length() { ensureOpen(); return length; }
    public int reservedTokens() { ensureOpen(); return state.reservedTokens(); }
    public void prepare(int nextLength) {
        ensureOpen();
        if (context.activeTape() != null) { throw new IllegalStateException("Paged attention is inference-only"); }
        if (nextLength <= length || nextLength > config.capacity()) { throw new IllegalArgumentException("Invalid next cache length"); }
        context.synchronize(); state.prepare(nextLength); prepared = nextLength;
    }
    public void commit(int nextLength) {
        ensureOpen();
        if (nextLength != prepared || nextLength <= length) { throw new IllegalStateException("Cache step was not prepared"); }
        length = nextLength;
    }
    public CachedAttention layer(int layer) {
        ensureOpen();
        if (layer < 0 || layer >= config.layers()) { throw new IllegalArgumentException("Cache layer out of range"); }
        return new CachedAttention() {
            public long length() { return PagedAttentionCache.this.length(); }
            public Tensor attend(Tensor query, Tensor key, Tensor value, Tensor allowed) {
                ensureOpen();
                if (context.activeTape() != null) { throw new IllegalStateException("Paged attention is inference-only"); }
                if (allowed != null) { throw new UnsupportedOperationException("Paged attention currently supports causal attention without an additional mask"); }
                for (Tensor tensor : List.of(query, key, value)) { context.requireContext(tensor); }
                int time = prepared - length;
                Shape q = Shape.of(1, config.heads(), time, config.headDim());
                Shape kv = Shape.of(1, config.kvHeads(), time, config.headDim());
                if (time <= 0 || query.dtype() != DType.FP32 || key.dtype() != DType.FP32 || value.dtype() != DType.FP32
                        || !query.shape().equals(q) || !key.shape().equals(kv) || !value.shape().equals(kv)) {
                    throw new IllegalArgumentException("Paged attention requires prepared FP32 [1,heads,time,headDim] inputs");
                }
                Tensor output = context.empty(DType.FP32, q);
                try {
                    TensorAccess.beforeWrite(output);
                    context.enqueueExternal(() -> state.attend(layer, length, TensorAccess.kernel(query), TensorAccess.kernel(key),
                            TensorAccess.kernel(value), TensorAccess.kernel(output)), List.of(query, key, value, output));
                    return output;
                } catch (RuntimeException | Error error) { output.close(); throw error; }
            }
        };
    }
    public void reset() {
        ensureOpen();
        if (context.activeTape() != null) { throw new IllegalStateException("Paged attention is inference-only"); }
        context.synchronize(); state.reset(); length = prepared = 0;
    }
    private void ensureOpen() { context.ensureOpen(); if (closed) { throw new IllegalStateException("Paged cache is closed"); } }
    @Override public void close() {
        context.checkThread();
        if (!closed) { closed = true; try { state.close(); } finally { context.releaseResource(this); } }
    }
}

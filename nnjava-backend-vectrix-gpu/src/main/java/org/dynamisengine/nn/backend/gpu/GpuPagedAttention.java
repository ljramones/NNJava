package org.dynamisengine.nn.backend.gpu;

import java.nio.ByteOrder;
import java.util.Set;
import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.api.compute.kit.StridedView;
import org.dynamisengine.gpu.primitives.AttentionF16;
import org.dynamisengine.gpu.primitives.Cast;
import org.dynamisengine.gpu.primitives.PagedKvCache;
import org.dynamisengine.gpu.primitives.Precision;
import org.dynamisengine.nn.backend.Completion;
import org.dynamisengine.nn.backend.KernelTensor;
import org.dynamisengine.nn.backend.PagedAttentionState;
import org.dynamisengine.nn.tensor.DType;

/** One sequence over an upstream all-layer FP16 page pool. No NNJava numerical kernels. */
final class GpuPagedAttention implements PagedAttentionState {
    private final GpuBackend backend;
    private final Config config;
    private final PagedKvCache cache;
    private final PagedKvCache.Sequence sequence;
    private final AttentionF16 attention;
    private final Cast cast;
    private final GpuBuffer lengths;
    private int valid;
    private boolean closed;

    GpuPagedAttention(GpuBackend backend, GpuDevice device, Config config) {
        this.backend = backend; this.config = config;
        PagedKvCache pool = null; AttentionF16 fused = null; Cast conversion = null; GpuBuffer lengthBuffer = null;
        try {
            pool = PagedKvCache.create(device, new PagedKvCache.Config(config.layers(), config.kvHeads(), config.headDim(),
                    config.pageSize(), Math.ceilDiv(config.capacity(), config.pageSize())));
            fused = AttentionF16.create(device, Precision.FP32);
            if (!fused.fusedSupports(shape(config, 1, config.capacity()))) { throw new UnsupportedOperationException("Device cannot execute fused paged attention"); }
            conversion = Cast.create(device);
            lengthBuffer = device.allocate(new BufferDescriptor(4, Set.of(BufferUsage.STORAGE, BufferUsage.MAP_WRITE), Placement.SHARED, "nnjava-kv-length"));
            cache = pool; attention = fused; cast = conversion; lengths = lengthBuffer; sequence = cache.newSequence();
        } catch (RuntimeException | Error error) {
            if (lengthBuffer != null) { lengthBuffer.close(); }
            if (conversion != null) { conversion.close(); }
            if (fused != null) { fused.close(); }
            if (pool != null) { pool.close(); }
            throw error;
        }
    }
    static AttentionF16.Shape shape(Config config, int queries, int keys) {
        return new AttentionF16.Shape(config.heads(), queries, keys, config.headDim(),
                (float) (1 / Math.sqrt(config.headDim())), queries > 1, config.kvHeads(), keys);
    }
    @Override public void prepare(int nextLength) {
        open();
        if (nextLength <= 0 || nextLength > config.capacity()) { throw new IllegalArgumentException("Invalid paged cache length"); }
        int reserved = Math.multiplyExact(Math.ceilDiv(nextLength, config.pageSize()), config.pageSize());
        if (reserved > sequence.tokens()) {
            backend.submitCommands(e -> cache.reserve(e, sequence, reserved - sequence.tokens())).await();
        }
        try (var mapping = lengths.map(Access.WRITE)) { mapping.bytes().order(ByteOrder.LITTLE_ENDIAN).putInt(nextLength); }
        valid = nextLength;
    }
    @Override public int reservedTokens() { open(); return sequence.tokens(); }
    @Override public Completion attend(int layer, int position, KernelTensor query, KernelTensor key, KernelTensor value, KernelTensor output) {
        open();
        int time = Math.toIntExact(query.layout().shape().dimension(2));
        if (position < 0 || position + time != valid) { throw new IllegalStateException("Paged attention step was not prepared"); }
        backend.recordingPositionLimit(sequence.tokens());
        return backend.submitCommands(e -> {
            try (var scratch = new GpuScratch(backend)) {
                var q = scratch.packed(e, query, true); var k = scratch.packed(e, key, true); var v = scratch.packed(e, value, true);
                var q16 = scratch.tensor(DType.FP16, q.layout().shape());
                var k16 = scratch.tensor(DType.FP16, k.layout().shape()); var v16 = scratch.tensor(DType.FP16, v.layout().shape());
                convert(e, q, q16); convert(e, k, k16); convert(e, v, v16);
                cache.write(e, layer, sequence, position, backend.storage(k16), backend.storage(v16), time,
                        Math.multiplyExact(time, config.headDim()), config.headDim());
                // One-token decode uses a stable page-sized key domain, restricted by the live length word.
                int keys = time == 1 ? sequence.tokens() : valid;
                attention.encode(e, backend.storage(q16), cache, layer, sequence, backend.storage(output), shape(config, time, keys),
                        time == 1 ? lengths : null, config.heads());
            }
        });
    }
    private void convert(GpuCommandEncoder e, KernelTensor from, KernelTensor to) {
        long n = from.layout().shape().elements();
        cast.cast(e, Cast.Type.FP32, Cast.Type.FP16, backend.storage(from), StridedView.contiguous(n), backend.storage(to), StridedView.contiguous(n));
    }
    @Override public void reset() { open(); cache.truncate(sequence, 0); valid = 0; }
    private void open() { if (closed) { throw new IllegalStateException("Paged cache is closed"); } }
    @Override public void close() {
        if (!closed) {
            closed = true;
            try { sequence.close(); cache.close(); } finally {
                try { lengths.close(); cast.close(); } finally { attention.close(); backend.releaseInferenceResource(this); }
            }
        }
    }
}

package org.dynamisengine.nn.backend;

/** Optional inference-only provider state. Inputs/output are FP32 [1,heads,time,headDim]. */
public interface PagedAttentionState extends AutoCloseable {
    record Config(int layers, int heads, int kvHeads, int headDim, int capacity, int pageSize) {
        public Config {
            if (layers <= 0 || heads <= 0 || kvHeads <= 0 || heads % kvHeads != 0 || headDim <= 0
                    || headDim % 2 != 0 || capacity <= 0 || pageSize <= 0 || pageSize % 32 != 0) {
                throw new IllegalArgumentException("Invalid paged attention dimensions");
            }
            Math.multiplyExact(Math.ceilDiv(capacity, pageSize), pageSize);
        }
    }
    /** Reserve pages and update the device's valid length before any layer of a step. */
    void prepare(int nextLength);
    int reservedTokens();
    Completion attend(int layer, int position, KernelTensor query, KernelTensor key, KernelTensor value, KernelTensor output);
    void reset();
    @Override void close();
}

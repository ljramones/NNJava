package org.dynamisengine.nn.transformer;

/** Session-only execution policy; never changes model weights or serialized architecture. */
public record InferenceOptions(CachePrecision cachePrecision, boolean recordDecode, int pageSize) {
    public enum CachePrecision { FP32_FIXED, FP16_PAGED }
    public InferenceOptions {
        java.util.Objects.requireNonNull(cachePrecision);
        if (pageSize <= 0 || pageSize % 32 != 0) { throw new IllegalArgumentException("Page size must be a positive multiple of 32"); }
        if (recordDecode && cachePrecision != CachePrecision.FP16_PAGED) {
            throw new IllegalArgumentException("Recorded decoding requires the explicit FP16 paged policy");
        }
    }
    public static InferenceOptions defaults() { return new InferenceOptions(CachePrecision.FP32_FIXED, false, 32); }
    public static InferenceOptions pagedFp16() { return new InferenceOptions(CachePrecision.FP16_PAGED, false, 32); }
    public InferenceOptions withRecordedDecode() { return new InferenceOptions(cachePrecision, true, pageSize); }
}

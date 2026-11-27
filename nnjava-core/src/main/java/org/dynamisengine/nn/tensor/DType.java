package org.dynamisengine.nn.tensor;

/**
 * Storage types. Only FP32 supports gradients; low-precision arithmetic is capability-gated.
 */
public enum DType {
    // Preserve the original ordinals for version-1 checkpoint compatibility.
    FP32(4), INT32(4), BOOL(1), FP16(2), BF16(2);

    private final int bytes;

    DType(int bytes) {
        this.bytes = bytes;
    }

    public int bytes() {
        return bytes;
    }

    public boolean isLowPrecision() {
        return this == FP16 || this == BF16;
    }
}

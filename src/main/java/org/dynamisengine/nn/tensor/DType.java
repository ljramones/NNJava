package org.dynamisengine.nn.tensor;

/** Storage types supported by the initial runtime. Only FP32 supports gradients. */
public enum DType {
    FP32(4), INT32(4), BOOL(1);

    private final int bytes;

    DType(int bytes) {
        this.bytes = bytes;
    }

    public int bytes() {
        return bytes;
    }
}

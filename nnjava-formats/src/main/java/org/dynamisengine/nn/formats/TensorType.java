package org.dynamisengine.nn.formats;

/** On-disk storage layouts; quantized values remain in their original GGML blocks. */
public enum TensorType {
    F32(1, 4), F16(1, 2), BF16(1, 2), I32(1, 4), BOOL(1, 1),
    Q4_0(32, 18), Q4_1(32, 20), Q5_0(32, 22), Q5_1(32, 24), Q8_0(32, 34),
    Q2_K(256, 84), Q3_K(256, 110), Q4_K(256, 144), Q5_K(256, 176), Q6_K(256, 210), Q8_K(256, 292);

    private final int blockElements, blockBytes;
    TensorType(int blockElements, int blockBytes) { this.blockElements = blockElements; this.blockBytes = blockBytes; }
    public int blockElements() { return blockElements; }
    public int blockBytes() { return blockBytes; }
    public boolean quantized() { return blockElements != 1; }
    public long byteSize(long elements) {
        if (elements < 0 || elements % blockElements != 0) { throw new IllegalArgumentException("Incomplete " + this + " block"); }
        return Math.multiplyExact(elements / blockElements, blockBytes);
    }

    static TensorType ggml(int id) {
        return switch (id) {
            case 0 -> F32; case 1 -> F16; case 2 -> Q4_0; case 3 -> Q4_1;
            case 6 -> Q5_0; case 7 -> Q5_1; case 8 -> Q8_0;
            case 10 -> Q2_K; case 11 -> Q3_K; case 12 -> Q4_K; case 13 -> Q5_K;
            case 14 -> Q6_K; case 15 -> Q8_K; case 26 -> I32; case 30 -> BF16;
            default -> throw new IllegalArgumentException("Unsupported GGML tensor type: " + id);
        };
    }
}

package org.dynamisengine.nn.inference.llama31metal;

import org.dynamisengine.gpu.primitives.QuantizedMatrix;

/** Conservative per-block bound on the magnitudes of a packed Q4_K or Q6_K matrix. */
final class ReconstructionRange {
    static boolean withinFiniteHalf(QuantizedMatrix w, byte[] data) {
        if (data.length != w.bytes()) throw new IllegalArgumentException("Incomplete packed matrix");
        int at = Math.toIntExact(w.parameterOffset());
        int end = Math.toIntExact(w.parameterOffset() + w.parameterBytes());
        switch (w.format()) {
            case Q4K -> {
                for (; at < end; at += 16) {
                    // |d * scale * code - dmin * min|, scale/min in [0,63], code in [0,15].
                    double bound = halfMagnitude(data, at) * (63 * 15) + halfMagnitude(data, at + 2) * 63;
                    if (!(bound <= 65504.0)) return false;
                }
            }
            case Q6K -> {
                for (; at < end; at += 20) {
                    // |d * signed scale * code|, |scale| <= 128 and |code| <= 32.
                    double bound = halfMagnitude(data, at) * (128 * 32);
                    if (!(bound <= 65504.0)) return false;
                }
            }
            default -> { return false; }
        }
        return true;
    }

    private static double halfMagnitude(byte[] bytes, int at) {
        short h = (short) ((bytes[at] & 255) | ((bytes[at + 1] & 255) << 8));
        return Math.abs((double) Float.float16ToFloat(h));
    }
}

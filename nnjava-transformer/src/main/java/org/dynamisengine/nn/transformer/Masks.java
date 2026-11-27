package org.dynamisengine.nn.transformer;

import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.*;

/**
 * Boolean attention masks use true for allowed positions.
 */
public final class Masks {
    private Masks() {
    }

    public static Tensor causal(ExecutionContext context, int queries, int keys, long queryOffset) {
        if (queries < 0 || keys < 0 || queryOffset < 0) {
            throw new IllegalArgumentException("Invalid causal mask dimensions");
        }
        Math.addExact(queryOffset, queries);
        boolean[] values = new boolean[Math.multiplyExact(queries, keys)];
        for (int q = 0; q < queries; q++) {
            for (int k = 0; k < keys; k++) {
                values[q * keys + k] = k <= queryOffset + q;
            }
        }
        return context.tensor(values, Shape.of(queries, keys));
    }

    /**
     * lengths[B] produces [B,1,1,keys] for broadcasting over heads and queries.
     */
    public static Tensor padding(Tensor lengths, int keys) {
        if (lengths.dtype() != DType.INT32 || lengths.shape().rank() != 1 || keys < 0) {
            throw new IllegalArgumentException("Padding lengths must be an integer vector");
        }
        int[] rows = lengths.toIntArray();
        boolean[] values = new boolean[Math.multiplyExact(rows.length, keys)];
        for (int b = 0; b < rows.length; b++) {
            if (rows[b] < 0 || rows[b] > keys) {
                throw new IllegalArgumentException("Padding length outside key capacity");
            }
            for (int k = 0; k < keys; k++) {
                values[b * keys + k] = k < rows[b];
            }
        }
        return lengths.context().tensor(values, Shape.of(rows.length, 1, 1, keys));
    }
}

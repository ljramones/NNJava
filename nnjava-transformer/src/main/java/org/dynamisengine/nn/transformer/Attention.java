package org.dynamisengine.nn.transformer;

import org.dynamisengine.nn.nn.Dropout;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.tensor.*;

/**
 * Differentiable scaled dot-product and grouped-query attention over [batch,heads,time,features].
 */
public final class Attention {
    private Attention() {
    }

    public static Tensor scaledDotProduct(Tensor query, Tensor key, Tensor value, Tensor allowed, boolean causal) {
        return scaledDotProduct(query, key, value, allowed, causal, 0, null);
    }

    public static Tensor scaledDotProduct(Tensor query, Tensor key, Tensor value, Tensor allowed,
                                          boolean causal, long queryOffset, Dropout dropout) {
        var context = query.context();
        context.requireContext(key);
        context.requireContext(value);
        Shape q = query.shape(), k = key.shape(), v = value.shape();
        if (q.rank() != 4 || k.rank() != 4 || v.rank() != 4 || q.dimension(1) <= 0 || k.dimension(1) <= 0
                || q.dimension(1) % k.dimension(1) != 0 || k.dimension(1) != v.dimension(1)
                || q.dimension(3) != k.dimension(3) || q.dimension(3) == 0 || k.dimension(2) != v.dimension(2) || k.dimension(2) == 0
                || v.dimension(3) == 0 || queryOffset < 0) {
            throw new IllegalArgumentException("Invalid Q/K/V attention dimensions");
        }
        try (Tensor expandedKey = heads(key, q.dimension(1)); Tensor expandedValue = heads(value, q.dimension(1));
             Tensor transposed = Ops.transpose(expandedKey); Tensor product = Ops.matmul(query, transposed);
             Tensor scores = Ops.scale(product, (float) (1 / Math.sqrt(q.dimension(3))))) {
            Tensor mask = allowed == null ? null : allowed.retain();
            try {
                if (mask != null) {
                    context.requireContext(mask);
                    if (mask.dtype() != DType.BOOL || !Shape.broadcast(mask.shape(), scores.shape()).equals(scores.shape())) {
                        throw new IllegalArgumentException("Attention mask must broadcast to scores");
                    }
                }
                if (causal) {
                    try (Tensor triangle = Masks.causal(context, Math.toIntExact(q.dimension(2)), Math.toIntExact(k.dimension(2)), queryOffset)) {
                        Tensor combined = mask == null ? triangle.retain() : Ops.and(mask, triangle);
                        if (mask != null) {
                            mask.close();
                        }
                        mask = combined;
                    }
                }
                try (Tensor excluded = context.full(Shape.of(), Float.NEGATIVE_INFINITY);
                     Tensor masked = mask == null ? scores.retain() : Ops.where(mask, scores, excluded);
                     Tensor probabilities = Ops.softmax(masked, -1);
                     Tensor effective = dropout == null ? probabilities.retain() : dropout.forward(probabilities)) {
                    return Ops.matmul(effective, expandedValue);
                }
            } finally {
                if (mask != null) {
                    mask.close();
                }
            }
        }
    }

    private static Tensor heads(Tensor input, long heads) {
        if (input.shape().dimension(1) == heads) {
            return input.retain();
        }
        int[] mapping = new int[Math.toIntExact(heads)];
        long group = heads / input.shape().dimension(1);
        for (int i = 0; i < mapping.length; i++) {
            mapping[i] = Math.toIntExact(i / group);
        }
        try (Tensor indices = input.context().tensor(mapping, Shape.of(heads))) {
            return Ops.gather(input, 1, indices);
        }
    }
}

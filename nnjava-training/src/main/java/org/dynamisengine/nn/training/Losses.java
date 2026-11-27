package org.dynamisengine.nn.training;

import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.tensor.Tensor;

/**
 * Differentiable loss compositions using the public operation API.
 */
public final class Losses {
    private Losses() {
    }

    public static Tensor crossEntropy(Tensor logits, Tensor targets) {
        return crossEntropy(logits, targets, Reduction.MEAN, -100);
    }

    public static Tensor crossEntropy(Tensor logits, Tensor targets, Reduction reduction, int ignoreIndex) {
        return org.dynamisengine.nn.ops.NeuralOps.crossEntropy(logits, targets, reduction.name().toLowerCase(java.util.Locale.ROOT), ignoreIndex);
    }

    public static Tensor meanSquaredError(Tensor prediction, Tensor target) {
        try (Tensor difference = Ops.subtract(prediction, target); Tensor squared = Ops.square(difference)) {
            return Ops.mean(squared);
        }
    }

    public enum Reduction {NONE, SUM, MEAN}
}

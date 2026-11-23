package org.dynamisengine.nn.training;

import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.tensor.Tensor;

/** Differentiable loss compositions using the public operation API. */
public final class Losses {
    private Losses() {
    }

    public static Tensor meanSquaredError(Tensor prediction, Tensor target) {
        try (Tensor difference = Ops.subtract(prediction, target); Tensor squared = Ops.square(difference)) {
            return Ops.mean(squared);
        }
    }
}

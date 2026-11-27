package org.dynamisengine.nn.training;

import org.dynamisengine.nn.tensor.Tensor;

/** Scalar mean loss and its denominator, used to aggregate unequal batches correctly. */
public interface TrainingObjective {
    /** Stable configuration identity for checkpoint compatibility. */
    String id();

    /** Returns an owned scalar mean loss; prediction and target are borrowed. */
    Tensor loss(Tensor prediction, Tensor target);

    /** Number of contributing target elements; zero skips the batch without updating. */
    long weight(Tensor target);

    /** Optional mean score with the same denominator; NaN means unavailable. */
    default double score(Tensor prediction, Tensor target) { return Double.NaN; }
}

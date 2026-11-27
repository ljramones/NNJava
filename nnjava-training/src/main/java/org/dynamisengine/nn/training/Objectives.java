package org.dynamisengine.nn.training;

import org.dynamisengine.nn.tensor.Tensor;

/** Standard supervised objectives for both neural networks and token prediction. */
public final class Objectives {
    private Objectives() { }
    public static TrainingObjective crossEntropy() { return crossEntropy(-100); }
    public static TrainingObjective crossEntropy(int ignoreIndex) { return new CrossEntropy(ignoreIndex); }
    public static TrainingObjective meanSquaredError() { return new MeanSquaredError(); }

    private record CrossEntropy(int ignoreIndex) implements TrainingObjective {
        public String id() { return "crossEntropy:" + ignoreIndex; }
        public Tensor loss(Tensor prediction, Tensor target) {
            return Losses.crossEntropy(prediction, target, Losses.Reduction.MEAN, ignoreIndex);
        }
        public long weight(Tensor target) {
            long count = 0;
            for (int value : target.toIntArray()) { if (value != ignoreIndex) { count++; } }
            return count;
        }
        public double score(Tensor prediction, Tensor target) { return Metrics.accuracy(prediction, target, ignoreIndex); }
    }

    private record MeanSquaredError() implements TrainingObjective {
        public String id() { return "meanSquaredError"; }
        public Tensor loss(Tensor prediction, Tensor target) {
            if (!prediction.shape().equals(target.shape())) {
                throw new IllegalArgumentException("MSE objective requires matching prediction and target shapes");
            }
            return Losses.meanSquaredError(prediction, target);
        }
        public long weight(Tensor target) { return target.shape().elements(); }
    }
}

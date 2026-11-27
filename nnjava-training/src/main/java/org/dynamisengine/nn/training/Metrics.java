package org.dynamisengine.nn.training;

import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;

public final class Metrics {
    private Metrics() {
    }

    public static double accuracy(Tensor logits, Tensor targets, int ignoreIndex) {
        try (Tensor predicted = Ops.argmax(logits, -1, false);
             Tensor ignore = targets.context().tensor(new int[]{ignoreIndex}, Shape.of());
             Tensor ignored = Ops.equal(targets, ignore); Tensor valid = Ops.not(ignored);
             Tensor correct = Ops.equal(predicted, targets); Tensor selected = Ops.and(correct, valid);
             Tensor counts = Ops.cast(selected, DType.FP32); Tensor validCounts = Ops.cast(valid, DType.FP32);
             Tensor numerator = Ops.sum(counts); Tensor denominator = Ops.sum(validCounts)) {
            return denominator.item() == 0 ? 0 : numerator.item() / denominator.item();
        }
    }
}

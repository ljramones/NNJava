package org.dynamisengine.nn.examples;

import org.dynamisengine.nn.nn.Linear;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;
import org.dynamisengine.nn.training.Losses;
import org.dynamisengine.nn.training.Sgd;

/** A small composition example: learn y = 3x + 2 through public framework APIs. */
public final class LinearRegressionExample {
    private LinearRegressionExample() {
    }

    public static void main(String[] args) {
        try (var context = ExecutionContext.cpu();
                var model = new Linear(context, 1, 1, 42);
                var optimizer = new Sgd(model.namedParameters().values(), 0.1f);
                var x = context.tensor(new float[] {-2, -1, 0, 1, 2}, Shape.of(5, 1));
                var y = context.tensor(new float[] {-4, -1, 2, 5, 8}, Shape.of(5, 1))) {
            float first = 0;
            float last = 0;
            for (int step = 0; step < 100; step++) {
                optimizer.zeroGrad();
                try (var tape = context.gradientTape();
                        Tensor prediction = model.forward(x);
                        Tensor loss = Losses.meanSquaredError(prediction, y)) {
                    last = loss.item();
                    if (step == 0) {
                        first = last;
                    }
                    tape.backward(loss);
                }
                optimizer.step();
            }
            try (Tensor weight = model.namedParameters().get("weight").value();
                    Tensor bias = model.namedParameters().get("bias").value()) {
                System.out.printf("MSE: %.6f -> %.9f; weight=%.6f, bias=%.6f%n",
                        first, last, weight.toFloatArray()[0], bias.toFloatArray()[0]);
            }
        }
    }
}

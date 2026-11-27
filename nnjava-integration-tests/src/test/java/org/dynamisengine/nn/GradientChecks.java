package org.dynamisengine.nn;

import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.assertEquals;

final class GradientChecks {
    private GradientChecks() {
    }

    static void check(ExecutionContext context, float[][] values, Shape[] shapes, Function<List<Tensor>, Tensor> function) {
        List<Tensor> parameters = new ArrayList<>();
        List<float[]> analytic = new ArrayList<>();
        try {
            for (int i = 0; i < values.length; i++) {
                parameters.add(context.tensor(values[i], shapes[i], true));
            }
            try (var tape = context.gradientTape(); Tensor loss = function.apply(parameters)) {
                tape.backward(loss);
            }
            for (Tensor parameter : parameters) {
                try (Tensor gradient = parameter.gradient()) {
                    analytic.add(gradient.toFloatArray());
                }
            }
        } finally {
            parameters.forEach(Tensor::close);
        }
        for (int p = 0; p < values.length; p++) {
            for (int i = 0; i < values[p].length; i++) {
                float old = values[p][i];
                float h = 0.002f;
                values[p][i] = old + h;
                float plus = evaluate(context, values, shapes, function);
                values[p][i] = old - h;
                float minus = evaluate(context, values, shapes, function);
                values[p][i] = old;
                float expected = (plus - minus) / (2 * h);
                assertEquals(expected, analytic.get(p)[i], 0.004 + Math.abs(expected) * 0.004,
                        "gradient parameter " + p + " element " + i);
            }
        }
    }

    private static float evaluate(ExecutionContext context, float[][] values, Shape[] shapes, Function<List<Tensor>, Tensor> function) {
        List<Tensor> arguments = new ArrayList<>();
        try {
            for (int i = 0; i < values.length; i++) {
                arguments.add(context.tensor(values[i], shapes[i]));
            }
            try (Tensor loss = function.apply(arguments)) {
                return loss.item();
            }
        } finally {
            arguments.forEach(Tensor::close);
        }
    }
}

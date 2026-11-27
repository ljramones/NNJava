package org.dynamisengine.nn;

import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;
import org.dynamisengine.nn.training.Losses;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class AutogradTest {
    private static double[] toDouble(float[] source) {
        double[] result = new double[source.length];
        for (int i = 0; i < source.length; i++) {
            result[i] = source[i];
        }
        return result;
    }

    private static double reference(double[] p) {
        double result = 0;
        for (int row = 0; row < 2; row++) {
            for (int column = 0; column < 2; column++) {
                double y = p[12 + column];
                for (int k = 0; k < 3; k++) {
                    y += p[row * 3 + k] * p[6 + k * 2 + column];
                }
                result += y * y / 4;
            }
        }
        for (int i = 0; i < 6; i++) {
            result += 0.1 * p[i] * p[i];
        }
        return result;
    }

    @Test
    void branchingAndBroadcastGradientsAccumulateAcrossPasses() {
        try (var context = ExecutionContext.cpu();
             var x = context.tensor(new float[]{1, 2, 3, 4, 5, 6}, Shape.of(2, 3), true);
             var bias = context.tensor(new float[]{2, 3, 4}, Shape.of(3), true)) {
            for (int pass = 1; pass <= 2; pass++) {
                try (var tape = context.gradientTape(); var squared = Ops.square(x);
                     var shifted = Ops.add(squared, bias); var total = Ops.sum(shifted)) {
                    tape.backward(total);
                }
                try (var dx = x.gradient(); var db = bias.gradient()) {
                    float[] expected = {2, 4, 6, 8, 10, 12};
                    for (int i = 0; i < expected.length; i++) {
                        expected[i] *= pass;
                    }
                    assertArrayEquals(expected, dx.toFloatArray());
                    assertArrayEquals(new float[]{2f * pass, 2f * pass, 2f * pass}, db.toFloatArray());
                }
            }
            x.clearGradient();
            bias.clearGradient();
            assertFalse(x.hasGradient());
            assertEquals(2, context.liveAllocations());
        }
    }

    @Test
    void matrixBroadcastAndBranchingGradientsMatchIndependentDoubleFiniteDifferences() {
        float[] all = {0.2f, -0.3f, 0.7f, 0.5f, -0.6f, 0.1f,
                0.4f, -0.2f, 0.8f, 0.3f, -0.5f, 0.6f, 0.1f, -0.4f};
        try (var context = ExecutionContext.cpu();
             var a = context.tensor(java.util.Arrays.copyOfRange(all, 0, 6), Shape.of(2, 3), true);
             var b = context.tensor(java.util.Arrays.copyOfRange(all, 6, 12), Shape.of(3, 2), true);
             var bias = context.tensor(java.util.Arrays.copyOfRange(all, 12, 14), Shape.of(2), true);
             var tape = context.gradientTape(); var product = Ops.matmul(a, b);
             var shifted = Ops.add(product, bias); var squared = Ops.square(shifted);
             var mean = Ops.mean(squared); var aSquared = Ops.square(a);
             var aSum = Ops.sum(aSquared); var penalty = Ops.scale(aSum, 0.1f);
             var loss = Ops.add(mean, penalty)) {
            assertEquals(reference(toDouble(all)), loss.item(), 1e-6);
            tape.backward(loss);
            try (var da = a.gradient(); var db = b.gradient(); var dc = bias.gradient()) {
                float[][] actual = {da.toFloatArray(), db.toFloatArray(), dc.toFloatArray()};
                int parameter = 0;
                for (float[] group : actual) {
                    for (float value : group) {
                        double[] point = toDouble(all);
                        double h = 1e-5;
                        point[parameter] += h;
                        double plus = reference(point);
                        point[parameter] -= 2 * h;
                        double expected = (plus - reference(point)) / (2 * h);
                        assertEquals(expected, value, 2e-5, "parameter " + parameter);
                        parameter++;
                    }
                }
            }
        }
    }

    @Test
    void viewsHaveBackwardRulesAndSavedHandlesSurviveCallerRelease() {
        try (var context = ExecutionContext.cpu();
             var leaf = context.tensor(new float[]{1, 2, 3, 4}, Shape.of(2, 2), true);
             var tape = context.gradientTape()) {
            Tensor transposed = Ops.transpose(leaf);
            Tensor packed = Ops.contiguous(transposed);
            Tensor reshaped = Ops.reshape(packed, Shape.of(4));
            Tensor squared = Ops.square(reshaped);
            try (Tensor total = Ops.sum(squared)) {
                transposed.close();
                packed.close();
                reshaped.close();
                squared.close();
                tape.backward(total);
            }
            try (var gradient = leaf.gradient()) {
                assertArrayEquals(new float[]{2, 4, 6, 8}, gradient.toFloatArray());
            }
            leaf.clearGradient();
            assertEquals(1, context.liveAllocations());
        }
    }

    @Test
    void rejectsMutationOfSavedStorageAndConsumesTapeOnce() {
        try (var context = ExecutionContext.cpu();
             var x = context.tensor(new float[]{3}, Shape.of(), true);
             var tape = context.gradientTape(); var loss = Ops.square(x)) {
            assertThrows(IllegalStateException.class, () -> context.fill(x, 99));
            try (var detached = x.detach()) {
                assertThrows(IllegalStateException.class, () -> context.fill(detached, 99));
            }
            tape.backward(loss);
            assertThrows(IllegalStateException.class, () -> tape.backward(loss));
            context.fill(x, 5);
            try (var gradient = x.gradient()) {
                assertEquals(6f, gradient.item());
            }
        }
    }

    @Test
    void inferenceDoesNotRecordAndForeignTapeHistoryRequiresDetach() {
        try (var context = ExecutionContext.cpu();
             var x = context.tensor(new float[]{2}, Shape.of(), true);
             var prediction = Ops.square(x)) {
            assertFalse(prediction.requiresGrad());
            Tensor history;
            try (var tape = context.gradientTape()) {
                history = Ops.square(x);
            }
            try (history; var tape = context.gradientTape()) {
                assertThrows(IllegalStateException.class, () -> Ops.square(history));
                try (var detached = history.detach(); var valid = Ops.add(x, detached)) {
                    tape.backward(valid);
                }
            }
            try (var gradient = x.gradient()) {
                assertEquals(1f, gradient.item());
            }
        }
    }

    @Test
    void nonScalarBackwardRequiresAnExplicitMatchingSeed() {
        try (var context = ExecutionContext.cpu();
             var x = context.tensor(new float[]{2, 3}, Shape.of(2), true);
             var seed = context.tensor(new float[]{1, 2}, Shape.of(2));
             var tape = context.gradientTape(); var result = Ops.square(x)) {
            assertThrows(IllegalArgumentException.class, () -> tape.backward(result));
            tape.backward(result, seed);
            try (var gradient = x.gradient()) {
                assertArrayEquals(new float[]{4, 12}, gradient.toFloatArray());
            }
        }
    }

    @Test
    void squaredErrorDifferentiatesBothPredictionAndTarget() {
        try (var context = ExecutionContext.cpu();
             var prediction = context.tensor(new float[]{2, 4}, Shape.of(2), true);
             var target = context.tensor(new float[]{0, 1}, Shape.of(2), true);
             var tape = context.gradientTape(); var loss = Losses.meanSquaredError(prediction, target)) {
            tape.backward(loss);
            try (var dp = prediction.gradient(); var dt = target.gradient()) {
                assertArrayEquals(new float[]{2, 3}, dp.toFloatArray());
                assertArrayEquals(new float[]{-2, -3}, dt.toFloatArray());
            }
        }
    }

    @Test
    void explicitBroadcastAndEmptySumHaveValidGradients() {
        try (var context = ExecutionContext.cpu();
             var leaf = context.tensor(new float[]{2, 3}, Shape.of(2), true);
             var empty = context.tensor(new float[0], Shape.of(0), true)) {
            try (var tape = context.gradientTape(); var expanded = Ops.broadcastTo(leaf, Shape.of(3, 2));
                 var total = Ops.sum(expanded)) {
                tape.backward(total);
            }
            try (var gradient = leaf.gradient()) {
                assertArrayEquals(new float[]{3, 3}, gradient.toFloatArray());
            }
            try (var tape = context.gradientTape(); var total = Ops.sum(empty)) {
                tape.backward(total);
            }
            try (var gradient = empty.gradient()) {
                assertEquals(Shape.of(0), gradient.shape());
                assertArrayEquals(new float[0], gradient.toFloatArray());
            }
        }
    }
}

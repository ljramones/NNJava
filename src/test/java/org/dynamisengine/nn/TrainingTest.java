package org.dynamisengine.nn;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;
import org.dynamisengine.nn.nn.Linear;
import org.dynamisengine.nn.nn.Parameter;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.training.Losses;
import org.dynamisengine.nn.training.Sgd;
import org.junit.jupiter.api.Test;

class TrainingTest {
    @Test
    void publicModulesAndOptimizerLearnLinearRegressionWithoutLeakingStepState() {
        try (var context = ExecutionContext.cpu(); var model = new Linear(context, 1, 1, 42);
                var optimizer = new Sgd(model.namedParameters().values(), 0.1f);
                var x = context.tensor(new float[] {-2, -1, 0, 1, 2}, Shape.of(5, 1));
                var y = context.tensor(new float[] {-4, -1, 2, 5, 8}, Shape.of(5, 1))) {
            int baseline = context.liveAllocations();
            float initial = 0;
            for (int step = 0; step < 100; step++) {
                optimizer.zeroGrad();
                assertEquals(baseline, context.liveAllocations());
                try (var tape = context.gradientTape(); var prediction = model.forward(x);
                        var loss = Losses.meanSquaredError(prediction, y)) {
                    if (step == 0) { initial = loss.item(); }
                    tape.backward(loss);
                }
                optimizer.step();
            }
            model.training(false);
            try (var prediction = model.forward(x); var loss = Losses.meanSquaredError(prediction, y);
                    var weight = model.namedParameters().get("weight").value();
                    var bias = model.namedParameters().get("bias").value()) {
                assertFalse(prediction.requiresGrad());
                assertTrue(loss.item() < initial * 1e-8f);
                assertEquals(3f, weight.toFloatArray()[0], 2e-5);
                assertEquals(2f, bias.toFloatArray()[0], 2e-5);
            }
            optimizer.zeroGrad();
            assertEquals(baseline, context.liveAllocations());
        }
    }

    @Test
    void sharedParametersAreUpdatedOnceAndEvaluationModeDoesNotDisableAutograd() {
        try (var context = ExecutionContext.cpu();
                var leaf = context.tensor(new float[] {2}, Shape.of(), true);
                var p = new Parameter("p", leaf); var alias = new Parameter("alias", leaf);
                var optimizer = new Sgd(List.of(p, alias), 0.25f)) {
            try (var tape = context.gradientTape(); var loss = Ops.square(leaf)) {
                assertThrows(IllegalStateException.class, optimizer::step);
                tape.backward(loss);
            }
            optimizer.step();
            assertEquals(1f, leaf.item());
        }
        try (var context = ExecutionContext.cpu(); var model = new Linear(context, 1, 1, 7);
                var input = context.full(Shape.of(1, 1), 1)) {
            model.training(false);
            try (var tape = context.gradientTape(); var output = model.forward(input); var total = Ops.sum(output)) {
                assertTrue(output.requiresGrad());
                tape.backward(total);
            }
        }
    }

    @Test
    void normalInitializationIsReproducible() {
        try (var context = ExecutionContext.cpu(); var a = new Linear(context, 3, 2, 55);
                var b = new Linear(context, 3, 2, 55);
                var av = a.namedParameters().get("weight").value();
                var bv = b.namedParameters().get("weight").value()) {
            assertArrayEquals(av.toFloatArray(), bv.toFloatArray());
        }
    }
}

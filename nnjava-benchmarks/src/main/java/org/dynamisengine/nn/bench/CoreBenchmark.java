package org.dynamisengine.nn.bench;

import java.util.concurrent.TimeUnit;

import org.openjdk.jmh.annotations.*;
import org.dynamisengine.nn.nn.*;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.*;
import org.dynamisengine.nn.training.*;

/**
 * End-to-end public API cost, including tensor ownership, host readback, and allocation.
 */
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
public class CoreBenchmark {
    @Param({"16", "64"})
    public int features;
    @Param({"4", "32"})
    public int batch;
    private ExecutionContext context;
    private Tensor input, weight, target;
    private Linear model;
    private Sgd optimizer;

    @Setup
    public void setup() {
        context = ExecutionContext.cpu();
        input = context.normal(Shape.of(batch, features), 1, .1f, false);
        weight = context.normal(Shape.of(features, features), 2, .1f, false);
        target = context.normal(Shape.of(batch, features), 3, .1f, false);
        model = new Linear(context, features, features, 4);
        optimizer = new Sgd(model.namedParameters(), 0);
    }

    @Benchmark
    public float matmulAndReadback() {
        try (Tensor output = Ops.matmul(input, weight)) {
            return output.toFloatArray()[0];
        }
    }

    @Benchmark
    public float forwardBackwardAndUpdate() {
        optimizer.zeroGrad();
        float value;
        try (var tape = context.gradientTape(); Tensor predicted = model.forward(input); Tensor loss = Losses.meanSquaredError(predicted, target)) {
            value = loss.item();
            tape.backward(loss);
        }
        optimizer.step();
        return value;
    }

    @TearDown
    public void teardown() {
        optimizer.close();
        model.close();
        input.close();
        weight.close();
        target.close();
        context.close();
    }
}

package org.dynamisengine.nn.nn;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;

/** Rank-two projection X * W^T + bias, with weights stored [outputs, inputs]. */
public final class Linear implements UnaryModule {
    private final ExecutionContext context;
    private final long inputFeatures;
    private final Parameter weight;
    private final Parameter bias;
    private final Map<String, Parameter> parameters;
    private boolean training = true;
    private boolean closed;

    public Linear(ExecutionContext context, long inputFeatures, long outputFeatures, long seed) {
        if (inputFeatures <= 0 || outputFeatures <= 0) {
            throw new IllegalArgumentException("Linear feature counts must be positive");
        }
        this.context = context;
        this.inputFeatures = inputFeatures;
        try (Tensor initialized = context.normal(Shape.of(outputFeatures, inputFeatures), seed,
                (float) Math.sqrt(1.0 / inputFeatures), true)) {
            weight = new Parameter("weight", initialized);
        }
        try (Tensor initialized = context.tensor(new float[Math.toIntExact(outputFeatures)],
                Shape.of(outputFeatures), true)) {
            bias = new Parameter("bias", initialized);
        } catch (RuntimeException error) {
            weight.close();
            throw error;
        }
        LinkedHashMap<String, Parameter> registered = new LinkedHashMap<>();
        registered.put("weight", weight);
        registered.put("bias", bias);
        parameters = Collections.unmodifiableMap(registered);
    }

    @Override
    public Tensor forward(Tensor input) {
        ensureOpen();
        context.requireContext(input);
        if (input.shape().rank() != 2 || input.shape().dimension(1) != inputFeatures) {
            throw new IllegalArgumentException("Linear input must be [batch, " + inputFeatures + "]");
        }
        try (Tensor w = weight.value(); Tensor b = bias.value(); Tensor transposed = Ops.transpose(w);
                Tensor projected = Ops.matmul(input, transposed)) {
            return Ops.add(projected, b);
        }
    }

    @Override public Map<String, Parameter> namedParameters() { ensureOpen(); return parameters; }
    @Override public boolean isTraining() { ensureOpen(); return training; }
    @Override public void training(boolean training) { ensureOpen(); this.training = training; }

    private void ensureOpen() {
        context.ensureOpen();
        if (closed) {
            throw new IllegalStateException("Linear module is closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            weight.close();
            bias.close();
            closed = true;
        }
    }
}

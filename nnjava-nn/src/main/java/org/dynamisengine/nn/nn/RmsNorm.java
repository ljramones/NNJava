package org.dynamisengine.nn.nn;

import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;

public final class RmsNorm extends AbstractModule implements UnaryModule {
    private final Parameter scale;
    private final float epsilon;

    public RmsNorm(ExecutionContext context, long features, float epsilon) {
        super(context);
        if (features <= 0 || !Float.isFinite(epsilon) || epsilon <= 0) {
            throw new IllegalArgumentException("Invalid normalization configuration");
        }
        this.epsilon = epsilon;
        scale = constantParameter("weight", Shape.of(features), 1);
    }

    @Override
    public Tensor forward(Tensor input) {
        ensureOpen();
        context.requireContext(input);
        try (Tensor weight = scale.value()) {
            return Ops.rmsNorm(input, weight, epsilon);
        }
    }

    @Override
    protected java.util.Map<String, String> localConfiguration() {
        return java.util.Map.of("epsilon", Float.toHexString(epsilon));
    }
}

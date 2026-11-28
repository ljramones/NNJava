package org.dynamisengine.nn.nn;

import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;

/**
 * Integer lookup with summed gradients for repeated token IDs.
 */
public final class Embedding extends AbstractModule implements UnaryModule {
    private final Parameter weight;

    public Embedding(ExecutionContext context, long vocabulary, long features, long seed) {
        this(context, vocabulary, features, seed, ParameterInitialization.DEFAULT);
    }

    public Embedding(ExecutionContext context, long vocabulary, long features, long seed, ParameterInitialization initialization) {
        super(context);
        if (vocabulary <= 0 || features <= 0) {
            throw new IllegalArgumentException("Positive embedding dimensions required");
        }
        try (Tensor initial = java.util.Objects.requireNonNull(initialization).normal(context, Shape.of(vocabulary, features), seed, .02f)) {
            weight = register("weight", initial);
        }
    }

    public Parameter weight() {
        ensureOpen();
        return weight;
    }

    @Override
    public Tensor forward(Tensor indices) {
        ensureOpen();
        context.requireContext(indices);
        try (Tensor values = weight.value()) {
            return Ops.gather(values, 0, indices);
        }
    }
}

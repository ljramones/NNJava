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
        super(context);
        if (vocabulary <= 0 || features <= 0) {
            throw new IllegalArgumentException("Positive embedding dimensions required");
        }
        try (Tensor initial = context.normal(Shape.of(vocabulary, features), seed, .02f, true)) {
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

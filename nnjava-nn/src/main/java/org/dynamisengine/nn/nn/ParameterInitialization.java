package org.dynamisengine.nn.nn;

import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;

/** Construction policy for normally random weights; bias and normalization defaults are unchanged. */
public enum ParameterInitialization {
    /** Existing seeded normal initialization, requiring provider support. */
    DEFAULT,
    /** Backend zero fill, intended for explicit weight loading; never falls back to a CPU initializer. */
    ZEROS;

    public Tensor normal(ExecutionContext context, Shape shape, long seed, float standardDeviation) {
        return this == DEFAULT ? context.normal(shape, seed, standardDeviation, true) : context.full(shape, 0, true);
    }

    public void validate(ExecutionContext context, String path, Shape shape) {
        ModelBuild.parameter(context, path, shape, this == DEFAULT);
    }
}

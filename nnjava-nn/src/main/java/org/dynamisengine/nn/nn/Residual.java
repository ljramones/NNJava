package org.dynamisengine.nn.nn;

import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Tensor;

public final class Residual extends AbstractModule implements UnaryModule {
    private final UnaryModule branch;

    public Residual(ExecutionContext context, UnaryModule branch) {
        super(context);
        this.branch = register("branch", branch);
    }

    @Override
    public Tensor forward(Tensor input) {
        ensureOpen();
        try (Tensor result = branch.forward(input)) {
            if (!result.shape().equals(input.shape())) {
                throw new IllegalArgumentException("Residual branch must preserve shape");
            }
            return Ops.add(input, result);
        }
    }
}

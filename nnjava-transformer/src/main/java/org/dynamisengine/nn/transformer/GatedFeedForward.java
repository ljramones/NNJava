package org.dynamisengine.nn.transformer;

import org.dynamisengine.nn.nn.AbstractModule;
import org.dynamisengine.nn.nn.Linear;
import org.dynamisengine.nn.nn.ParameterInitialization;
import org.dynamisengine.nn.nn.UnaryModule;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Tensor;

/**
 * Reusable SwiGLU component with separate gate, expansion, and output projections.
 */
public final class GatedFeedForward extends AbstractModule implements UnaryModule {
    private final Linear gate, up, down;

    public GatedFeedForward(ExecutionContext context, int features, int hidden, long seed) {
        this(context, features, hidden, seed, ParameterInitialization.DEFAULT);
    }

    public GatedFeedForward(ExecutionContext context, int features, int hidden, long seed, ParameterInitialization initialization) {
        super(context);
        try {
            gate = register("gate", new Linear(context, features, hidden, seed, false, initialization));
            up = register("up", new Linear(context, features, hidden, seed + 1, false, initialization));
            down = register("down", new Linear(context, hidden, features, seed + 2, false, initialization));
        } catch (RuntimeException | Error error) {
            close();
            throw error;
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        ensureOpen();
        try (Tensor projected = gate.forward(input); Tensor active = Ops.silu(projected);
             Tensor expanded = up.forward(input); Tensor gated = Ops.multiply(active, expanded)) {
            return down.forward(gated);
        }
    }
}

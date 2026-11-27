package org.dynamisengine.nn.transformer;

import org.dynamisengine.nn.nn.*;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Tensor;

/** Reusable SwiGLU component with separate gate, expansion, and output projections. */
public final class GatedFeedForward extends AbstractModule implements UnaryModule {
    private final Linear gate, up, down;
    public GatedFeedForward(ExecutionContext context, int features, int hidden, long seed) {
        super(context);
        gate = register("gate", new Linear(context, features, hidden, seed, false));
        up = register("up", new Linear(context, features, hidden, seed + 1, false));
        down = register("down", new Linear(context, hidden, features, seed + 2, false));
    }
    @Override public Tensor forward(Tensor input) {
        ensureOpen();
        try (Tensor projected = gate.forward(input); Tensor active = Ops.silu(projected);
                Tensor expanded = up.forward(input); Tensor gated = Ops.multiply(active, expanded)) { return down.forward(gated); }
    }
}

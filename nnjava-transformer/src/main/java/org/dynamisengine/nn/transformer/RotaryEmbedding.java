package org.dynamisengine.nn.transformer;

import org.dynamisengine.nn.nn.AbstractModule;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Tensor;

/**
 * Explicit positions or a shared sequence offset, with half or interleaved feature pairing.
 */
public final class RotaryEmbedding extends AbstractModule {
    private final double base;
    private final boolean half;

    public RotaryEmbedding(ExecutionContext context, double base, boolean half) {
        super(context);
        if (!Double.isFinite(base) || base <= 0) {
            throw new IllegalArgumentException("Invalid rotary base");
        }
        this.base = base;
        this.half = half;
    }

    public Tensor forward(Tensor input, long offset) {
        ensureOpen();
        context.requireContext(input);
        return Ops.rope(input, offset, base, half);
    }

    /**
     * Position IDs broadcast to all axes preceding the feature axis, e.g. [time] or [batch,1,time].
     */
    public Tensor forward(Tensor input, Tensor positions) {
        ensureOpen();
        context.requireContext(input);
        return Ops.rope(input, positions, base, half);
    }

    @Override
    protected java.util.Map<String, String> localConfiguration() {
        return java.util.Map.of("base", Double.toHexString(base), "half", Boolean.toString(half));
    }
}

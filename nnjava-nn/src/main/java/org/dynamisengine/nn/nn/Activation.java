package org.dynamisengine.nn.nn;

import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Tensor;

public final class Activation extends AbstractModule implements UnaryModule {
    private final Kind kind;

    public Activation(ExecutionContext context, Kind kind) {
        super(context);
        this.kind = java.util.Objects.requireNonNull(kind);
    }

    public static Activation relu(ExecutionContext c) {
        return new Activation(c, Kind.RELU);
    }

    public static Activation tanh(ExecutionContext c) {
        return new Activation(c, Kind.TANH);
    }

    public static Activation gelu(ExecutionContext c) {
        return new Activation(c, Kind.GELU);
    }

    public static Activation silu(ExecutionContext c) {
        return new Activation(c, Kind.SILU);
    }

    @Override
    protected java.util.Map<String, String> localConfiguration() {
        return java.util.Map.of("kind", kind.name());
    }

    @Override
    public Tensor forward(Tensor input) {
        ensureOpen();
        context.requireContext(input);
        return switch (kind) {
            case RELU -> Ops.relu(input);
            case TANH -> Ops.tanh(input);
            case SIGMOID -> Ops.sigmoid(input);
            case GELU -> Ops.gelu(input);
            case SILU -> Ops.silu(input);
        };
    }

    public enum Kind {RELU, TANH, SIGMOID, GELU, SILU}
}

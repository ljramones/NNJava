package org.dynamisengine.nn.nn;

import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;

/**
 * Projection over the last axis; weights are [outputs, inputs].
 */
public final class Linear extends AbstractModule implements UnaryModule {
    public static Builder builder() { return new Builder(); }

    public record Definition(String name, long inputFeatures, long outputFeatures, boolean bias, long seed)
            implements ModelDefinition<Linear> {
        public Definition {
            ModelBuild.name(name);
            if (inputFeatures <= 0 || outputFeatures <= 0) { throw new IllegalArgumentException("Positive feature counts required"); }
            Shape.of(outputFeatures, inputFeatures);
        }
        @Override public void validate(ExecutionContext context) {
            validate(context, ParameterInitialization.DEFAULT);
        }
        @Override public void validate(ExecutionContext context, ParameterInitialization initialization) {
            ModelBuild.context(context);
            java.util.Objects.requireNonNull(initialization).validate(context, name + ".weight", Shape.of(outputFeatures, inputFeatures));
            if (bias) { ModelBuild.parameter(context, name + ".bias", Shape.of(outputFeatures), false); }
            ModelBuild.require(context, name, org.dynamisengine.nn.tensor.DType.FP32,
                    org.dynamisengine.nn.backend.Operation.COPY, org.dynamisengine.nn.backend.Operation.FILL,
                    org.dynamisengine.nn.backend.Operation.MATMUL, org.dynamisengine.nn.backend.Operation.SUM_TO);
            if (bias) { ModelBuild.require(context, name, org.dynamisengine.nn.tensor.DType.FP32, org.dynamisengine.nn.backend.Operation.ADD); }
        }
        @Override public Linear build(ExecutionContext context) { validate(context); return new Linear(context, inputFeatures, outputFeatures, seed, bias); }
        @Override public Linear build(ExecutionContext context, ParameterInitialization initialization) {
            validate(context, initialization); return new Linear(context, inputFeatures, outputFeatures, seed, bias, initialization);
        }
        @Override public org.dynamisengine.nn.model.ComponentSpec specification() {
            return new org.dynamisengine.nn.model.ComponentSpec("nnjava.linear", 1, name,
                    java.util.Map.of("inputs", Long.toString(inputFeatures), "outputs", Long.toString(outputFeatures),
                            "bias", Boolean.toString(bias), "seed", Long.toString(seed)), java.util.List.of());
        }
        public static Definition fromSpec(org.dynamisengine.nn.model.ComponentSpec spec) {
            spec.require("nnjava.linear", 1, "inputs", "outputs", "bias", "seed");
            if (!spec.children().isEmpty()) { throw new IllegalArgumentException("Linear definition cannot have children"); }
            return new Definition(spec.name(), spec.longValue("inputs"), spec.longValue("outputs"), spec.booleanValue("bias"), spec.longValue("seed"));
        }
    }

    public static final class Builder {
        private String name = "linear";
        private long inputs, outputs, seed;
        private boolean bias = true;
        private Builder() { }
        public Builder name(String value) { ModelBuild.name(value); name = value; return this; }
        public Builder inputFeatures(long value) { inputs = value; return this; }
        public Builder outputFeatures(long value) { outputs = value; return this; }
        public Builder bias(boolean value) { bias = value; return this; }
        public Builder seed(long value) { seed = value; return this; }
        public Definition define() { return new Definition(name, inputs, outputs, bias, seed); }
        public Linear build(ExecutionContext context) { return define().build(context); }
        public Linear build(ExecutionContext context, ParameterInitialization initialization) { return define().build(context, initialization); }
    }
    private final long inputFeatures;
    private final long outputFeatures;
    private final Parameter weight;
    private final Parameter bias;

    public Linear(ExecutionContext context, long inputs, long outputs, long seed) {
        this(context, inputs, outputs, seed, true);
    }

    public Linear(ExecutionContext context, long inputs, long outputs, long seed, boolean withBias) {
        this(context, inputs, outputs, seed, withBias, ParameterInitialization.DEFAULT);
    }

    public Linear(ExecutionContext context, long inputs, long outputs, long seed, boolean withBias, ParameterInitialization initialization) {
        super(context);
        if (inputs <= 0 || outputs <= 0) {
            throw new IllegalArgumentException("Positive feature counts required");
        }
        inputFeatures = inputs;
        outputFeatures = outputs;
        try (Tensor initial = java.util.Objects.requireNonNull(initialization).normal(context, Shape.of(outputs, inputs), seed, (float) Math.sqrt(1.0 / inputs))) {
            weight = register("weight", initial);
        }
        try {
            bias = withBias ? constantParameter("bias", Shape.of(outputs), 0) : null;
        } catch (RuntimeException error) {
            close();
            throw error;
        }
    }

    /**
     * Share an existing trainable matrix, retaining independent ownership of its value.
     */
    public Linear(ExecutionContext context, Parameter sharedWeight, boolean withBias) {
        super(context);
        try (Tensor shared = sharedWeight.value()) {
            context.requireContext(shared);
            if (shared.shape().rank() != 2) {
                throw new IllegalArgumentException("Shared weight must be a matrix");
            }
            outputFeatures = shared.shape().dimension(0);
            inputFeatures = shared.shape().dimension(1);
            if (outputFeatures == 0 || inputFeatures == 0) {
                throw new IllegalArgumentException("Empty shared weight");
            }
            weight = register("weight", shared);
        }
        try {
            bias = withBias ? constantParameter("bias", Shape.of(outputFeatures), 0) : null;
        } catch (RuntimeException error) {
            close();
            throw error;
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        ensureOpen();
        context.requireContext(input);
        Shape shape = input.shape();
        if (shape.rank() < 1 || shape.dimension(shape.rank() - 1) != inputFeatures) {
            throw new IllegalArgumentException("Linear last axis must equal " + inputFeatures);
        }
        Shape result = shape.prefix(shape.rank() - 1).append(outputFeatures);
        try (Tensor packed = Ops.contiguous(input); Tensor matrix = Ops.reshape(packed, Shape.of(shape.elements() / inputFeatures, inputFeatures));
             Tensor w = weight.value(); Tensor transposed = Ops.transpose(w); Tensor projected = Ops.matmul(matrix, transposed);
             Tensor shaped = Ops.reshape(projected, result)) {
            if (bias == null) {
                return shaped.retain();
            }
            try (Tensor b = bias.value()) {
                return Ops.add(shaped, b);
            }
        }
    }
}

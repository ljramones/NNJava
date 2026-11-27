package org.dynamisengine.nn.nn;

import java.util.List;
import java.util.Map;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.model.ComponentSpec;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;

/** Built-in immutable layer definitions. Feature dimensions are inferred by the network. */
public final class Layers {
    private Layers() { }
    public static Dense dense(long outputs) { return new Dense(outputs, true, null); }
    public static Nonlinearity activation(Activation.Kind kind) { return new Nonlinearity(kind); }
    public static DropoutSpec dropout(float probability) { return new DropoutSpec(probability); }
    public static Normalization layerNorm(float epsilon) { return new Normalization(NormalizationKind.LAYER, epsilon); }
    public static Normalization rmsNorm(float epsilon) { return new Normalization(NormalizationKind.RMS, epsilon); }
    public static Flatten flatten() { return new Flatten(); }
    public static Reshape reshape(long... dimensions) { return new Reshape(ModelBuild.sampleShape(Shape.of(dimensions))); }

    public static LayerDefinition fromSpec(ComponentSpec spec) {
        if (!spec.children().isEmpty()) { throw new IllegalArgumentException(spec.name() + ": layer has unexpected children"); }
        return switch (spec.type()) {
            case "nnjava.dense" -> {
                spec.require("nnjava.dense", 1, "outputs", "bias", "activation");
                String activation = spec.attribute("activation");
                yield new Dense(spec.longValue("outputs"), spec.booleanValue("bias"),
                        activation.equals("NONE") ? null : Activation.Kind.valueOf(activation));
            }
            case "nnjava.activation" -> {
                spec.require("nnjava.activation", 1, "kind");
                yield activation(Activation.Kind.valueOf(spec.attribute("kind")));
            }
            case "nnjava.dropout" -> {
                spec.require("nnjava.dropout", 1, "probability");
                yield dropout(spec.floatValue("probability"));
            }
            case "nnjava.normalization" -> {
                spec.require("nnjava.normalization", 1, "kind", "epsilon");
                yield new Normalization(NormalizationKind.valueOf(spec.attribute("kind")), spec.floatValue("epsilon"));
            }
            case "nnjava.flatten" -> { spec.require("nnjava.flatten", 1); yield flatten(); }
            case "nnjava.reshape" -> {
                spec.require("nnjava.reshape", 1, "shape");
                yield new Reshape(parseShape(spec.attribute("shape")));
            }
            default -> throw new IllegalArgumentException("Unknown layer type: " + spec.type());
        };
    }

    static String encodeShape(Shape shape) {
        return java.util.Arrays.stream(shape.dimensions()).mapToObj(Long::toString).collect(java.util.stream.Collectors.joining(","));
    }
    static Shape parseShape(String value) {
        return ModelBuild.sampleShape(Shape.of(java.util.Arrays.stream(value.split(",", -1)).mapToLong(Long::parseLong).toArray()));
    }
    private static long features(Shape shape) { ModelBuild.sampleShape(shape); return shape.dimension(shape.rank() - 1); }
    private static ComponentSpec spec(String type, String name, Map<String, String> attributes) {
        return new ComponentSpec(type, 1, name, attributes, List.of());
    }

    public record Dense(long outputs, boolean bias, Activation.Kind activation) implements LayerDefinition {
        public Dense { if (outputs <= 0) { throw new IllegalArgumentException("Dense outputs must be positive"); } }
        public Dense bias(boolean value) { return new Dense(outputs, value, activation); }
        public Dense activation(Activation.Kind value) { return new Dense(outputs, bias, java.util.Objects.requireNonNull(value)); }
        @Override public Shape outputShape(Shape input) { features(input); return input.replace(input.rank() - 1, outputs); }
        @Override public void validate(ExecutionContext context, Shape input, String path) {
            outputShape(input);
            ModelBuild.parameter(context, path + ".linear.weight", Shape.of(outputs, features(input)), true);
            if (bias) { ModelBuild.parameter(context, path + ".linear.bias", Shape.of(outputs), false); }
            ModelBuild.require(context, path, DType.FP32, Operation.COPY, Operation.FILL, Operation.MATMUL, Operation.SUM_TO);
            if (bias) { ModelBuild.require(context, path, DType.FP32, Operation.ADD); }
            if (activation != null) { new Nonlinearity(activation).validate(context, outputShape(input), path + ".activation"); }
        }
        @Override public UnaryModule build(ExecutionContext context, Shape input, long seed) {
            validate(context, input, "dense");
            return new DenseLayer(context, features(input), this, seed);
        }
        @Override public ComponentSpec specification(String name) {
            return spec("nnjava.dense", name, Map.of("outputs", Long.toString(outputs), "bias", Boolean.toString(bias),
                    "activation", activation == null ? "NONE" : activation.name()));
        }
    }

    public record Nonlinearity(Activation.Kind kind) implements LayerDefinition {
        public Nonlinearity { java.util.Objects.requireNonNull(kind); }
        @Override public Shape outputShape(Shape input) { return ModelBuild.sampleShape(input); }
        @Override public void validate(ExecutionContext context, Shape input, String path) {
            outputShape(input);
            ModelBuild.require(context, path, DType.FP32, Operation.valueOf(kind.name()), Operation.valueOf(kind.name() + "_BACKWARD"));
        }
        @Override public UnaryModule build(ExecutionContext context, Shape input, long seed) { return new Activation(context, kind); }
        @Override public ComponentSpec specification(String name) { return spec("nnjava.activation", name, Map.of("kind", kind.name())); }
    }

    public record DropoutSpec(float probability) implements LayerDefinition {
        public DropoutSpec {
            if (!Float.isFinite(probability) || probability < 0 || probability >= 1) { throw new IllegalArgumentException("Invalid dropout probability"); }
        }
        @Override public Shape outputShape(Shape input) { return ModelBuild.sampleShape(input); }
        @Override public void validate(ExecutionContext context, Shape input, String path) {
            outputShape(input);
            if (probability > 0) { ModelBuild.require(context, path, DType.FP32, Operation.DROPOUT, Operation.MULTIPLY); }
        }
        @Override public UnaryModule build(ExecutionContext context, Shape input, long seed) { return new Dropout(context, probability, seed); }
        @Override public ComponentSpec specification(String name) { return spec("nnjava.dropout", name, Map.of("probability", Float.toHexString(probability))); }
    }

    public record Normalization(NormalizationKind kind, float epsilon) implements LayerDefinition {
        public Normalization {
            java.util.Objects.requireNonNull(kind);
            if (!Float.isFinite(epsilon) || epsilon <= 0) { throw new IllegalArgumentException("Invalid normalization epsilon"); }
        }
        @Override public Shape outputShape(Shape input) { return ModelBuild.sampleShape(input); }
        @Override public void validate(ExecutionContext context, Shape input, String path) {
            ModelBuild.parameter(context, path + ".weight", Shape.of(features(input)), false);
            ModelBuild.require(context, path, DType.FP32, Operation.FILL, Operation.COPY, Operation.NORM_BACKWARD,
                    kind == NormalizationKind.RMS ? Operation.RMS_NORM : Operation.LAYER_NORM);
        }
        @Override public UnaryModule build(ExecutionContext context, Shape input, long seed) {
            return kind == NormalizationKind.RMS ? new RmsNorm(context, features(input), epsilon) : new LayerNorm(context, features(input), epsilon);
        }
        @Override public ComponentSpec specification(String name) {
            return spec("nnjava.normalization", name, Map.of("kind", kind.name(), "epsilon", Float.toHexString(epsilon)));
        }
    }

    public record Flatten() implements LayerDefinition {
        @Override public Shape outputShape(Shape input) { return Shape.of(ModelBuild.sampleShape(input).elements()); }
        @Override public void validate(ExecutionContext context, Shape input, String path) { ModelBuild.require(context, path, DType.FP32, Operation.COPY); }
        @Override public UnaryModule build(ExecutionContext context, Shape input, long seed) { return new LayoutLayer(context, input.rank(), outputShape(input)); }
        @Override public ComponentSpec specification(String name) { return spec("nnjava.flatten", name, Map.of()); }
    }

    public record Reshape(Shape shape) implements LayerDefinition {
        public Reshape { ModelBuild.sampleShape(shape); }
        @Override public Shape outputShape(Shape input) {
            if (input.elements() != shape.elements()) { throw new IllegalArgumentException("Reshape changes the sample element count"); }
            return shape;
        }
        @Override public void validate(ExecutionContext context, Shape input, String path) { outputShape(input); ModelBuild.require(context, path, DType.FP32, Operation.COPY); }
        @Override public UnaryModule build(ExecutionContext context, Shape input, long seed) { return new LayoutLayer(context, input.rank(), shape); }
        @Override public ComponentSpec specification(String name) { return spec("nnjava.reshape", name, Map.of("shape", encodeShape(shape))); }
    }

    private static final class DenseLayer extends AbstractModule implements UnaryModule {
        private final Linear linear;
        private final Activation activation;
        DenseLayer(ExecutionContext context, long inputs, Dense definition, long seed) {
            super(context);
            try {
                linear = register("linear", new Linear(context, inputs, definition.outputs, seed, definition.bias));
                activation = definition.activation == null ? null : register("activation", new Activation(context, definition.activation));
            } catch (RuntimeException | Error error) { close(); throw error; }
        }
        @Override public Tensor forward(Tensor input) {
            ensureOpen();
            try (Tensor projected = linear.forward(input)) { return activation == null ? projected.retain() : activation.forward(projected); }
        }
    }

    private static final class LayoutLayer extends AbstractModule implements UnaryModule {
        private final int inputRank;
        private final Shape target;
        LayoutLayer(ExecutionContext context, int inputRank, Shape target) { super(context); this.inputRank = inputRank; this.target = target; }
        @Override public Tensor forward(Tensor input) {
            ensureOpen(); context.requireContext(input);
            Shape shape = input.shape().prefix(input.shape().rank() - inputRank).append(target.dimensions());
            try (Tensor packed = Ops.contiguous(input)) { return Ops.reshape(packed, shape); }
        }
        @Override protected Map<String, String> localConfiguration() { return Map.of("shape", encodeShape(target), "inputRank", Integer.toString(inputRank)); }
    }
}

package org.dynamisengine.nn.nn;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import org.dynamisengine.nn.model.ComponentSpec;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;

/** Named sequential composition built from a reusable, allocation-free definition. */
public final class NeuralNetwork extends AbstractModule implements UnaryModule {
    private final Definition definition;
    private final List<UnaryModule> layers = new ArrayList<>();

    private NeuralNetwork(ExecutionContext context, Definition definition, ParameterInitialization initialization) {
        super(context);
        this.definition = definition;
        Shape shape = definition.inputShape;
        try {
            for (NamedLayer layer : definition.layers) {
                UnaryModule module = layer.layer.build(context, shape, ModelBuild.seed(definition.seed, layer.name), initialization);
                try { register(layer.name, module); }
                catch (RuntimeException | Error error) { module.close(); throw error; }
                layers.add(module);
                shape = layer.layer.outputShape(shape);
            }
        } catch (RuntimeException | Error error) { close(); throw error; }
    }

    public static Builder builder() { return new Builder(); }
    public Definition definition() { ensureOpen(); return definition; }

    @Override public Tensor forward(Tensor input) {
        ensureOpen(); context.requireContext(input);
        if (input.dtype() != DType.FP32) { throw new IllegalArgumentException(definition.name + ": input must be FP32"); }
        definition.outputShape(input.shape());
        Tensor value = input.retain();
        try {
            for (UnaryModule layer : layers) {
                Tensor next = layer.forward(value); value.close(); value = next;
            }
            return value;
        } catch (RuntimeException | Error error) { value.close(); throw error; }
    }

    @Override protected Map<String, String> localConfiguration() { return ModelBuild.configuration(definition.specification()); }

    public record NamedLayer(String name, LayerDefinition layer) {
        public NamedLayer { ModelBuild.name(name); java.util.Objects.requireNonNull(layer); }
    }

    public record Definition(String name, Shape inputShape, long seed, List<NamedLayer> layers)
            implements ModelDefinition<NeuralNetwork> {
        public Definition {
            ModelBuild.name(name); ModelBuild.sampleShape(inputShape); layers = List.copyOf(layers);
            if (layers.isEmpty()) { throw new IllegalArgumentException(name + ": at least one layer is required"); }
            var names = new HashSet<String>();
            Shape shape = inputShape;
            for (NamedLayer layer : layers) {
                if (!names.add(layer.name)) { throw new IllegalArgumentException(name + ": duplicate layer " + layer.name); }
                try { shape = ModelBuild.sampleShape(layer.layer.outputShape(shape)); }
                catch (IllegalArgumentException | ArithmeticException error) {
                    throw new IllegalArgumentException(name + "." + layer.name + ": " + error.getMessage(), error);
                }
            }
        }

        public Shape outputShape() {
            Shape shape = inputShape;
            for (NamedLayer layer : layers) { shape = layer.layer.outputShape(shape); }
            return shape;
        }

        public Shape outputShape(Shape actualInput) {
            int prefix = actualInput.rank() - inputShape.rank();
            if (prefix < 0) { throw new IllegalArgumentException(name + ": input rank is too small"); }
            for (int axis = 0; axis < inputShape.rank(); axis++) {
                if (actualInput.dimension(prefix + axis) != inputShape.dimension(axis)) {
                    throw new IllegalArgumentException(name + ": expected trailing sample shape " + inputShape + ", got " + actualInput);
                }
            }
            return actualInput.prefix(prefix).append(outputShape().dimensions());
        }

        @Override public void validate(ExecutionContext context) {
            validate(context, ParameterInitialization.DEFAULT);
        }
        @Override public void validate(ExecutionContext context, ParameterInitialization initialization) {
            ModelBuild.context(context);
            java.util.Objects.requireNonNull(initialization);
            Shape shape = inputShape;
            for (NamedLayer layer : layers) {
                layer.layer.validate(context, shape, name + "." + layer.name, initialization);
                shape = layer.layer.outputShape(shape);
            }
        }

        @Override public NeuralNetwork build(ExecutionContext context) { return build(context, ParameterInitialization.DEFAULT); }
        @Override public NeuralNetwork build(ExecutionContext context, ParameterInitialization initialization) {
            validate(context, initialization); return new NeuralNetwork(context, this, initialization);
        }
        @Override public ComponentSpec specification() {
            return new ComponentSpec("nnjava.sequential", 1, name,
                    Map.of("inputShape", Layers.encodeShape(inputShape), "seed", Long.toString(seed)),
                    layers.stream().map(layer -> layer.layer.specification(layer.name)).toList());
        }

        public static Definition fromSpec(ComponentSpec spec) { return fromSpec(spec, Map.of()); }
        /** Explicit decoders restore custom layer types; no global registry or reflective class loading. */
        public static Definition fromSpec(ComponentSpec spec, Map<String, Function<ComponentSpec, LayerDefinition>> custom) {
            spec.require("nnjava.sequential", 1, "inputShape", "seed");
            Map<String, Function<ComponentSpec, LayerDefinition>> factories = Map.copyOf(custom);
            List<NamedLayer> layers = spec.children().stream().map(child -> new NamedLayer(child.name(),
                    factories.containsKey(child.type()) ? factories.get(child.type()).apply(child) : Layers.fromSpec(child))).toList();
            return new Definition(spec.name(), Layers.parseShape(spec.attribute("inputShape")), spec.longValue("seed"), layers);
        }
    }

    /** Mutable/reusable builder; define() takes an immutable snapshot. */
    public static final class Builder {
        private String name = "network";
        private Shape inputShape;
        private long seed;
        private final List<NamedLayer> layers = new ArrayList<>();
        private Builder() { }
        public Builder name(String value) { ModelBuild.name(value); name = value; return this; }
        public Builder inputFeatures(long value) { return inputShape(value); }
        public Builder inputShape(long... values) { inputShape = ModelBuild.sampleShape(Shape.of(values)); return this; }
        public Builder seed(long value) { seed = value; return this; }
        public Builder layer(String name, LayerDefinition layer) { layers.add(new NamedLayer(name, layer)); return this; }
        public Builder dense(String name, long outputs) { return layer(name, Layers.dense(outputs)); }
        public Builder dense(String name, long outputs, Activation.Kind activation) { return layer(name, Layers.dense(outputs).activation(activation)); }
        public Builder dense(long outputs) { return dense(nextName("dense"), outputs); }
        public Builder dense(long outputs, Activation.Kind activation) { return dense(nextName("dense"), outputs, activation); }
        public Builder activation(String name, Activation.Kind kind) { return layer(name, Layers.activation(kind)); }
        public Builder dropout(String name, float probability) { return layer(name, Layers.dropout(probability)); }
        public Builder layerNorm(String name, float epsilon) { return layer(name, Layers.layerNorm(epsilon)); }
        public Builder rmsNorm(String name, float epsilon) { return layer(name, Layers.rmsNorm(epsilon)); }
        public Builder flatten(String name) { return layer(name, Layers.flatten()); }
        public Builder conv2d(String name,long outputs,int kernel) { return layer(name,Layers.conv2d(outputs,kernel)); }
        public Builder conv2d(String name,long outputs,int kernel,Activation.Kind activation) { return layer(name,Layers.conv2d(outputs,kernel).activation(activation)); }
        public Builder conv2d(String name,Layers.Convolution definition) { return layer(name,definition); }
        public Builder maxPool2d(String name,int window) { return layer(name,Layers.maxPool2d(window)); }
        public Builder avgPool2d(String name,int window) { return layer(name,Layers.avgPool2d(window)); }
        public Builder rnn(String name,long hidden) { return layer(name,Layers.rnn(hidden)); }
        public Builder rnn(String name,long hidden,SequenceOutput output) { return layer(name,Layers.rnn(hidden).output(output)); }
        public Builder lstm(String name,long hidden) { return layer(name,Layers.lstm(hidden)); }
        public Builder lstm(String name,long hidden,SequenceOutput output) { return layer(name,Layers.lstm(hidden).output(output)); }
        public Builder gru(String name,long hidden) { return layer(name,Layers.gru(hidden)); }
        public Builder gru(String name,long hidden,SequenceOutput output) { return layer(name,Layers.gru(hidden).output(output)); }
        public Builder reshape(String name, long... shape) { return layer(name, Layers.reshape(shape)); }
        public Definition define() {
            if (inputShape == null) { throw new IllegalArgumentException(name + ": input shape is required"); }
            return new Definition(name, inputShape, seed, layers);
        }
        public NeuralNetwork build(ExecutionContext context) { return define().build(context); }
        public NeuralNetwork build(ExecutionContext context, ParameterInitialization initialization) { return define().build(context, initialization); }
        private String nextName(String prefix) {
            var used = layers.stream().map(NamedLayer::name).collect(java.util.stream.Collectors.toSet());
            int ordinal = 0;
            while (used.contains(prefix + ordinal)) { ordinal++; }
            return prefix + ordinal;
        }
    }
}

package org.dynamisengine.nn.transformer;

import java.util.List;
import java.util.Map;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.model.ComponentSpec;
import org.dynamisengine.nn.nn.*;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.runtime.CachedAttention;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;

/** Configurable pre-normalized decoder block using the public attention and layer APIs. */
public final class TransformerBlock extends AbstractModule implements UnaryModule {
    private final Config config;
    private final UnaryModule attentionNorm, feedForwardNorm, feedForward;
    private final MultiHeadAttention attention;
    private final Dropout attentionDropout, feedForwardDropout;

    public TransformerBlock(ExecutionContext context, Config config, long seed) {
        this(context, config, seed, ParameterInitialization.DEFAULT);
    }

    public TransformerBlock(ExecutionContext context, Config config, long seed, ParameterInitialization initialization) {
        super(context);
        this.config = java.util.Objects.requireNonNull(config);
        config.validate(context, "block", initialization);
        try {
            int width = config.attention.features();
            attentionNorm = register("attentionNorm", normalization(context, width, config));
            attention = register("attention", new MultiHeadAttention(context, config.attention, ModelBuild.seed(seed, "attention"), initialization));
            feedForwardNorm = register("feedForwardNorm", normalization(context, width, config));
            feedForward = register("feedForward", config.feedForward == FeedForwardKind.SWIGLU
                    ? new GatedFeedForward(context, width, config.hiddenWidth, ModelBuild.seed(seed, "feedForward"), initialization)
                    : new PlainFeedForward(context, width, config.hiddenWidth, config.feedForward, ModelBuild.seed(seed, "feedForward"), initialization));
            attentionDropout = register("attentionDropout", new Dropout(context, config.residualDropout, ModelBuild.seed(seed, "attentionDropout")));
            feedForwardDropout = register("feedForwardDropout", new Dropout(context, config.residualDropout, ModelBuild.seed(seed, "feedForwardDropout")));
        } catch (RuntimeException | Error error) { close(); throw error; }
    }

    private static UnaryModule normalization(ExecutionContext context, int width, Config config) {
        return config.normalization == NormalizationKind.RMS ? new RmsNorm(context, width, config.epsilon)
                : new LayerNorm(context, width, config.epsilon);
    }

    @Override public Tensor forward(Tensor input) { return forward(input, null); }
    public Tensor forward(Tensor input, Tensor allowed) { return compute(input, allowed, null); }
    public Tensor decode(Tensor input, KvCache cache, Tensor allowed) {
        return decode(input, (CachedAttention) cache, allowed);
    }
    public Tensor decode(Tensor input, CachedAttention cache, Tensor allowed) {
        ensureOpen();
        if (isTraining() || context.activeTape() != null) { throw new IllegalStateException("Cached decoding requires evaluation mode without a gradient tape"); }
        return compute(input, allowed, java.util.Objects.requireNonNull(cache));
    }

    private Tensor compute(Tensor input, Tensor allowed, CachedAttention cache) {
        ensureOpen(); context.requireContext(input);
        try (Tensor normalized = attentionNorm.forward(input);
                Tensor attended = cache == null ? attention.forward(normalized, allowed, true) : attention.decode(normalized, cache, allowed);
                Tensor droppedAttention = attentionDropout.forward(attended); Tensor residual = Ops.add(input, droppedAttention);
                Tensor feedInput = feedForwardNorm.forward(residual); Tensor feed = feedForward.forward(feedInput);
                Tensor droppedFeed = feedForwardDropout.forward(feed)) {
            return Ops.add(residual, droppedFeed);
        }
    }

    @Override protected Map<String, String> localConfiguration() { return ModelBuild.configuration(config.specification("block")); }

    public record Config(MultiHeadAttention.Config attention, int hiddenWidth, FeedForwardKind feedForward,
                         NormalizationKind normalization, float epsilon, float residualDropout) {
        public Config {
            java.util.Objects.requireNonNull(attention); java.util.Objects.requireNonNull(feedForward); java.util.Objects.requireNonNull(normalization);
            if (hiddenWidth <= 0) { throw new IllegalArgumentException("Feed-forward hidden width must be positive"); }
            new Layers.Normalization(normalization, epsilon); new Layers.DropoutSpec(residualDropout);
        }

        public void validate(ExecutionContext context, String path) {
            validate(context, path, ParameterInitialization.DEFAULT);
        }

        public void validate(ExecutionContext context, String path, ParameterInitialization initialization) {
            ModelBuild.context(context);
            java.util.Objects.requireNonNull(initialization);
            int width = attention.features();
            int kvWidth = Math.multiplyExact(attention.kvHeads(), width / attention.heads());
            initialization.validate(context, path + ".attention.query.weight", Shape.of(width, width));
            initialization.validate(context, path + ".attention.key.weight", Shape.of(kvWidth, width));
            initialization.validate(context, path + ".attention.value.weight", Shape.of(kvWidth, width));
            initialization.validate(context, path + ".attention.output.weight", Shape.of(width, width));
            initialization.validate(context, path + ".feedForward.up.weight", Shape.of(hiddenWidth, width));
            initialization.validate(context, path + ".feedForward.down.weight", Shape.of(width, hiddenWidth));
            if (feedForward == FeedForwardKind.SWIGLU) { initialization.validate(context, path + ".feedForward.gate.weight", Shape.of(hiddenWidth, width)); }
            if (attention.bias()) {
                for (String projection : List.of("query", "key", "value", "output")) {
                    ModelBuild.parameter(context, path + ".attention." + projection + ".bias",
                            Shape.of(projection.equals("key") || projection.equals("value") ? kvWidth : width), false);
                }
            }
            ModelBuild.require(context, path, DType.FP32, Operation.COPY, Operation.FILL, Operation.MATMUL, Operation.ADD,
                    Operation.SCALE, Operation.MULTIPLY, Operation.SUM_TO, Operation.SOFTMAX, Operation.SOFTMAX_BACKWARD, Operation.WHERE);
            ModelBuild.require(context, path + ".mask", DType.BOOL, Operation.AND);
            if (attention.heads() != attention.kvHeads()) {
                ModelBuild.require(context, path + ".attention", DType.FP32, Operation.GATHER, Operation.SCATTER_ADD);
            }
            if (attention.rotary()) { ModelBuild.require(context, path + ".rotary", DType.FP32, Operation.ROPE, Operation.ROPE_BACKWARD); }
            new Layers.Normalization(normalization, epsilon).validate(context, Shape.of(width), path + ".normalization");
            new Layers.DropoutSpec(attention.dropout()).validate(context, Shape.of(width), path + ".attention.dropout");
            new Layers.DropoutSpec(residualDropout).validate(context, Shape.of(width), path + ".residualDropout");
            Activation.Kind activation = feedForward == FeedForwardKind.SWIGLU ? Activation.Kind.SILU : Activation.Kind.valueOf(feedForward.name());
            new Layers.Nonlinearity(activation).validate(context, Shape.of(hiddenWidth), path + ".feedForward.activation");
        }

        public ComponentSpec specification(String name) {
            var settings = new ComponentSpec("nnjava.selfAttention", 1, "attention", Map.of(
                    "features", Integer.toString(attention.features()), "queryHeads", Integer.toString(attention.heads()),
                    "keyValueHeads", Integer.toString(attention.kvHeads()), "bias", Boolean.toString(attention.bias()),
                    "rotary", Boolean.toString(attention.rotary()), "rotaryBase", Double.toHexString(attention.rotaryBase()),
                    "dropout", Float.toHexString(attention.dropout())), List.of());
            return new ComponentSpec("nnjava.transformerBlock", 1, name, Map.of("hiddenWidth", Integer.toString(hiddenWidth),
                    "feedForward", feedForward.name(), "normalization", normalization.name(), "epsilon", Float.toHexString(epsilon),
                    "residualDropout", Float.toHexString(residualDropout)), List.of(settings));
        }

        public static Config fromSpec(ComponentSpec spec) {
            spec.require("nnjava.transformerBlock", 1, "hiddenWidth", "feedForward", "normalization", "epsilon", "residualDropout");
            if (spec.children().size() != 1 || !spec.children().getFirst().name().equals("attention")) {
                throw new IllegalArgumentException(spec.name() + ": expected one attention definition");
            }
            ComponentSpec a = spec.children().getFirst();
            a.require("nnjava.selfAttention", 1, "features", "queryHeads", "keyValueHeads", "bias", "rotary", "rotaryBase", "dropout");
            if (!a.children().isEmpty()) { throw new IllegalArgumentException("Attention settings cannot have children"); }
            var attention = new MultiHeadAttention.Config(a.intValue("features"), a.intValue("queryHeads"), a.intValue("keyValueHeads"),
                    a.booleanValue("bias"), a.booleanValue("rotary"), a.doubleValue("rotaryBase"), a.floatValue("dropout"));
            return new Config(attention, spec.intValue("hiddenWidth"), FeedForwardKind.valueOf(spec.attribute("feedForward")),
                    NormalizationKind.valueOf(spec.attribute("normalization")), spec.floatValue("epsilon"), spec.floatValue("residualDropout"));
        }
    }

    private static final class PlainFeedForward extends AbstractModule implements UnaryModule {
        private final Linear up, down;
        private final Activation activation;
        PlainFeedForward(ExecutionContext context, int width, int hidden, FeedForwardKind kind, long seed, ParameterInitialization initialization) {
            super(context);
            try {
                up = register("up", new Linear(context, width, hidden, seed, false, initialization));
                activation = register("activation", new Activation(context, Activation.Kind.valueOf(kind.name())));
                down = register("down", new Linear(context, hidden, width, seed + 1, false, initialization));
            } catch (RuntimeException | Error error) { close(); throw error; }
        }
        @Override public Tensor forward(Tensor input) {
            ensureOpen();
            try (Tensor projected = up.forward(input); Tensor active = activation.forward(projected)) { return down.forward(active); }
        }
    }
}

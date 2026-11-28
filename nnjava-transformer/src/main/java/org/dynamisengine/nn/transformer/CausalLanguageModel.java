package org.dynamisengine.nn.transformer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.model.ComponentSpec;
import org.dynamisengine.nn.nn.*;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;

/** Generic causal decoder composition. Applications supply token IDs, masks, data, and recipes. */
public final class CausalLanguageModel extends AbstractModule implements UnaryModule {
    private final Definition definition;
    private final Embedding embedding;
    private final Dropout embeddingDropout;
    private final List<TransformerBlock> blocks = new ArrayList<>();
    private final UnaryModule finalNorm;
    private final Linear output;

    private CausalLanguageModel(ExecutionContext context, Definition definition) {
        super(context);
        this.definition = definition;
        try {
            embedding = register("embedding", new Embedding(context, definition.vocabularySize, definition.modelWidth,
                    ModelBuild.seed(definition.seed, "embedding")));
            embeddingDropout = register("embeddingDropout", new Dropout(context, definition.embeddingDropout,
                    ModelBuild.seed(definition.seed, "embeddingDropout")));
            for (int i = 0; i < definition.blockCount; i++) {
                String name = "block" + i;
                blocks.add(register(name, new TransformerBlock(context, definition.block, ModelBuild.seed(definition.seed, name))));
            }
            finalNorm = !definition.finalNormalization ? null : register("finalNorm",
                    definition.block.normalization() == NormalizationKind.RMS
                            ? new RmsNorm(context, definition.modelWidth, definition.block.epsilon())
                            : new LayerNorm(context, definition.modelWidth, definition.block.epsilon()));
            output = register("output", definition.tieInputOutputEmbeddings
                    ? new Linear(context, embedding.weight(), false)
                    : new Linear(context, definition.modelWidth, definition.vocabularySize, ModelBuild.seed(definition.seed, "output"), false));
        } catch (RuntimeException | Error error) { close(); throw error; }
    }

    public static Builder builder() { return new Builder(); }
    public Definition definition() { ensureOpen(); return definition; }
    /** Borrowed execution context, for adapters using this model's public inference API. */
    public ExecutionContext executionContext() { ensureOpen(); return context; }
    @Override public Tensor forward(Tensor tokens) { return forward(tokens, null); }
    public Tensor forward(Tensor tokens, Tensor allowed) {
        validateTokens(tokens);
        validateMask(allowed, tokens.shape().dimension(0), tokens.shape().dimension(1), tokens.shape().dimension(1));
        return compute(tokens, allowed, null);
    }

    /** The caller explicitly selects evaluation mode; this method never changes shared model mode. */
    public InferenceSession newSession(int batchSize) { return new InferenceSession(this, batchSize); }

    private void validateTokens(Tensor tokens) {
        ensureOpen(); context.requireContext(tokens);
        if (tokens.dtype() != DType.INT32 || tokens.shape().rank() != 2
                || tokens.shape().dimension(0) <= 0 || tokens.shape().dimension(1) <= 0
                || tokens.shape().dimension(1) > definition.contextLength) {
            throw new IllegalArgumentException(definition.name + ": tokens must be nonempty INT32 [batch,time] within contextLength");
        }
    }

    private void validateMask(Tensor allowed, long batch, long queries, long keys) {
        if (allowed == null) { return; }
        context.requireContext(allowed);
        Shape scores = Shape.of(batch, definition.block.attention().heads(), queries, keys);
        if (allowed.dtype() != DType.BOOL || !Shape.broadcast(allowed.shape(), scores).equals(scores)) {
            throw new IllegalArgumentException(definition.name + ": allowed mask must be BOOL and broadcast to " + scores);
        }
    }

    private Tensor compute(Tensor tokens, Tensor allowed, KvCache[] caches) {
        try (Tensor embedded = embedding.forward(tokens)) {
            Tensor hidden = embeddingDropout.forward(embedded);
            try {
                for (int i = 0; i < blocks.size(); i++) {
                    Tensor next = caches == null ? blocks.get(i).forward(hidden, allowed) : blocks.get(i).decode(hidden, caches[i], allowed);
                    hidden.close(); hidden = next;
                }
                try (Tensor normalized = finalNorm == null ? hidden.retain() : finalNorm.forward(hidden)) { return output.forward(normalized); }
            } finally { hidden.close(); }
        }
    }

    @Override protected Map<String, String> localConfiguration() { return ModelBuild.configuration(definition.specification()); }

    public record Definition(String name, int vocabularySize, int modelWidth, int blockCount, int contextLength,
                             TransformerBlock.Config block, boolean finalNormalization,
                             boolean tieInputOutputEmbeddings, float embeddingDropout, long seed)
            implements ModelDefinition<CausalLanguageModel> {
        public Definition {
            ModelBuild.name(name); java.util.Objects.requireNonNull(block);
            if (vocabularySize <= 0 || modelWidth <= 0 || blockCount <= 0 || contextLength <= 0) {
                throw new IllegalArgumentException(name + ": vocabularySize, modelWidth, blockCount and contextLength must be positive");
            }
            if (block.attention().features() != modelWidth) { throw new IllegalArgumentException(name + ": attention width differs from modelWidth"); }
            new Layers.DropoutSpec(embeddingDropout);
        }

        @Override public void validate(ExecutionContext context) {
            ModelBuild.context(context);
            ModelBuild.parameter(context, name + ".embedding.weight", Shape.of(vocabularySize, modelWidth), true);
            ModelBuild.require(context, name + ".embedding", DType.FP32, Operation.GATHER, Operation.SCATTER_ADD, Operation.COPY, Operation.FILL);
            for (int i = 0; i < blockCount; i++) { block.validate(context, name + ".block" + i); }
            new Layers.DropoutSpec(embeddingDropout).validate(context, Shape.of(modelWidth), name + ".embeddingDropout");
            if (finalNormalization) { new Layers.Normalization(block.normalization(), block.epsilon()).validate(context, Shape.of(modelWidth), name + ".finalNorm"); }
            if (!tieInputOutputEmbeddings) { ModelBuild.parameter(context, name + ".output.weight", Shape.of(vocabularySize, modelWidth), true); }
        }
        @Override public CausalLanguageModel build(ExecutionContext context) { validate(context); return new CausalLanguageModel(context, this); }
        @Override public ComponentSpec specification() {
            return new ComponentSpec("nnjava.causalLanguageModel", 1, name, Map.of("vocabularySize", Integer.toString(vocabularySize),
                    "modelWidth", Integer.toString(modelWidth), "blockCount", Integer.toString(blockCount), "contextLength", Integer.toString(contextLength),
                    "finalNormalization", Boolean.toString(finalNormalization), "tieInputOutputEmbeddings", Boolean.toString(tieInputOutputEmbeddings),
                    "embeddingDropout", Float.toHexString(embeddingDropout), "seed", Long.toString(seed)), List.of(block.specification("block")));
        }
        public static Definition fromSpec(ComponentSpec spec) {
            spec.require("nnjava.causalLanguageModel", 1, "vocabularySize", "modelWidth", "blockCount", "contextLength",
                    "finalNormalization", "tieInputOutputEmbeddings", "embeddingDropout", "seed");
            if (spec.children().size() != 1 || !spec.children().getFirst().name().equals("block")) {
                throw new IllegalArgumentException(spec.name() + ": expected one decoder block template");
            }
            return new Definition(spec.name(), spec.intValue("vocabularySize"), spec.intValue("modelWidth"), spec.intValue("blockCount"),
                    spec.intValue("contextLength"), TransformerBlock.Config.fromSpec(spec.children().getFirst()),
                    spec.booleanValue("finalNormalization"), spec.booleanValue("tieInputOutputEmbeddings"),
                    spec.floatValue("embeddingDropout"), spec.longValue("seed"));
        }
    }

    public static final class Builder {
        private String name = "decoder";
        private int vocabularySize, modelWidth = 128, blockCount = 2, contextLength = 512;
        private long seed;
        private boolean finalNormalization = true, tieEmbeddings = true;
        private float epsilon = 1e-5f, embeddingDropout, residualDropout;
        private NormalizationKind normalization = NormalizationKind.RMS;
        private final AttentionBuilder attention = new AttentionBuilder();
        private final FeedForwardBuilder feedForward = new FeedForwardBuilder();
        private Builder() { }
        public Builder name(String value) { ModelBuild.name(value); name = value; return this; }
        public Builder vocabularySize(int value) { vocabularySize = value; return this; }
        public Builder modelWidth(int value) { modelWidth = value; return this; }
        public Builder blockCount(int value) { blockCount = value; return this; }
        public Builder contextLength(int value) { contextLength = value; return this; }
        public Builder seed(long value) { seed = value; return this; }
        public Builder attention(Consumer<AttentionBuilder> settings) { settings.accept(attention); return this; }
        public Builder feedForward(Consumer<FeedForwardBuilder> settings) { settings.accept(feedForward); return this; }
        public Builder normalization(NormalizationKind value) { normalization = java.util.Objects.requireNonNull(value); return this; }
        public Builder normalizationEpsilon(float value) { epsilon = value; return this; }
        public Builder finalNormalization(boolean value) { finalNormalization = value; return this; }
        public Builder tieInputOutputEmbeddings(boolean value) { tieEmbeddings = value; return this; }
        public Builder embeddingDropout(float value) { embeddingDropout = value; return this; }
        public Builder residualDropout(float value) { residualDropout = value; return this; }
        public Definition define() {
            MultiHeadAttention.Config config;
            try {
                config = new MultiHeadAttention.Config(modelWidth, attention.queryHeads,
                        attention.keyValueHeads == 0 ? attention.queryHeads : attention.keyValueHeads,
                        attention.bias, attention.rotary, attention.rotaryBase, attention.dropout);
            } catch (IllegalArgumentException error) { throw new IllegalArgumentException(name + ".attention: check head divisibility, rotary dimensions, and dropout", error); }
            int hidden = feedForward.hiddenWidth == 0 ? Math.multiplyExact(modelWidth, 4) : feedForward.hiddenWidth;
            var block = new TransformerBlock.Config(config, hidden, feedForward.kind, normalization, epsilon, residualDropout);
            return new Definition(name, vocabularySize, modelWidth, blockCount, contextLength, block, finalNormalization,
                    tieEmbeddings, embeddingDropout, seed);
        }
        public CausalLanguageModel build(ExecutionContext context) { return define().build(context); }
    }

    public static final class AttentionBuilder {
        private int queryHeads = 4, keyValueHeads;
        private boolean bias, rotary = true;
        private double rotaryBase = 10_000;
        private float dropout;
        private AttentionBuilder() { }
        public AttentionBuilder queryHeads(int value) { queryHeads = value; return this; }
        public AttentionBuilder keyValueHeads(int value) {
            if (value <= 0) { throw new IllegalArgumentException("keyValueHeads must be positive"); }
            keyValueHeads = value; return this;
        }
        public AttentionBuilder bias(boolean value) { bias = value; return this; }
        public AttentionBuilder rotaryPositions() { rotary = true; return this; }
        public AttentionBuilder rotaryPositions(boolean value) { rotary = value; return this; }
        public AttentionBuilder rotaryBase(double value) { rotaryBase = value; return this; }
        public AttentionBuilder dropout(float value) { dropout = value; return this; }
    }

    public static final class FeedForwardBuilder {
        private int hiddenWidth;
        private FeedForwardKind kind = FeedForwardKind.SWIGLU;
        private FeedForwardBuilder() { }
        public FeedForwardBuilder hiddenWidth(int value) {
            if (value <= 0) { throw new IllegalArgumentException("hiddenWidth must be positive"); }
            hiddenWidth = value; return this;
        }
        public FeedForwardBuilder swiGlu() { kind = FeedForwardKind.SWIGLU; return this; }
        public FeedForwardBuilder gelu() { kind = FeedForwardKind.GELU; return this; }
        public FeedForwardBuilder silu() { kind = FeedForwardKind.SILU; return this; }
        public FeedForwardBuilder relu() { kind = FeedForwardKind.RELU; return this; }
    }

    /** Owns caches and borrows model weights. Failed compute invalidates this session. */
    public static final class InferenceSession implements AutoCloseable {
        private final CausalLanguageModel model;
        private final int batchSize;
        private final KvCache[] caches;
        private final Parameter[] parameters;
        private final long[] versions;
        private long length;
        private boolean closed, failed;

        private InferenceSession(CausalLanguageModel model, int batchSize) {
            this.model = model;
            model.ensureOpen();
            if (batchSize <= 0) { throw new IllegalArgumentException("Session batch size must be positive"); }
            this.batchSize = batchSize;
            requireInference();
            var config = model.definition.block.attention();
            Shape shape = Shape.of(batchSize, config.kvHeads(), model.definition.contextLength, config.features() / config.heads());
            ModelBuild.parameter(model.context, model.definition.name + ".cache", shape, false);
            caches = new KvCache[model.blocks.size()];
            parameters = model.namedParameters().values().toArray(Parameter[]::new);
            versions = new long[parameters.length];
            refreshVersions();
            try {
                for (int i = 0; i < caches.length; i++) {
                    caches[i] = new KvCache(model.context, batchSize, config.kvHeads(), model.definition.contextLength, config.features() / config.heads());
                }
            } catch (RuntimeException | Error error) {
                for (KvCache cache : caches) { if (cache != null) { cache.close(); } }
                throw error;
            }
        }

        public long length() { ensureOpen(); return length; }
        public long capacity() { ensureOpen(); return model.definition.contextLength; }
        public Tensor prefill(Tensor tokens) { return prefill(tokens, null); }
        public Tensor prefill(Tensor tokens, Tensor allowed) {
            ensureOpen();
            if (length != 0) { throw new IllegalStateException("Reset the session before prefill"); }
            return append(tokens, allowed);
        }
        public Tensor decode(Tensor tokens) { return decode(tokens, null); }
        public Tensor decode(Tensor tokens, Tensor allowed) {
            ensureOpen();
            if (length == 0) { throw new IllegalStateException("Prefill the session before decode"); }
            return append(tokens, allowed);
        }

        private Tensor append(Tensor tokens, Tensor allowed) {
            ensureOpen(); requireInference();
            model.validateTokens(tokens);
            if (tokens.shape().dimension(0) != batchSize) { throw new IllegalArgumentException("Session batch size mismatch"); }
            long next = Math.addExact(length, tokens.shape().dimension(1));
            if (next > capacity()) { throw new IllegalArgumentException("Session context capacity exceeded"); }
            model.validateMask(allowed, batchSize, tokens.shape().dimension(1), next);
            for (int i = 0; i < parameters.length; i++) {
                if (parameters[i].mutationVersion() != versions[i]) {
                    throw new IllegalStateException("Model weights changed; reset the session before continuing");
                }
            }
            try {
                Tensor result = model.compute(tokens, allowed, caches);
                length = next;
                return result;
            } catch (RuntimeException | Error error) { failed = true; throw error; }
        }

        public void reset() {
            ensureOpen(); requireInference();
            for (KvCache cache : caches) { cache.reset(); }
            length = 0; refreshVersions();
        }
        private void refreshVersions() { for (int i = 0; i < parameters.length; i++) { versions[i] = parameters[i].mutationVersion(); } }
        private void requireInference() {
            if (model.isTraining() || model.context.activeTape() != null) {
                throw new IllegalStateException("Inference sessions require model.training(false) and no active gradient tape");
            }
        }
        private void ensureOpen() {
            model.ensureOpen();
            if (closed || failed) { throw new IllegalStateException(failed ? "Inference session failed; close it and create another" : "Inference session is closed"); }
        }
        @Override public void close() {
            model.context.checkThread();
            if (!closed) { for (KvCache cache : caches) { cache.close(); } closed = true; }
        }
    }
}

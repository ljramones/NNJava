package org.dynamisengine.nn.transformer;

import org.dynamisengine.nn.nn.AbstractModule;
import org.dynamisengine.nn.nn.Dropout;
import org.dynamisengine.nn.nn.Linear;
import org.dynamisengine.nn.nn.ParameterInitialization;
import org.dynamisengine.nn.nn.UnaryModule;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;

/**
 * Configurable self-attention building block; inputs and outputs are [batch,time,features].
 */
public final class MultiHeadAttention extends AbstractModule implements UnaryModule {
    private final Config config;
    private final Linear query, key, value, output;
    private final Dropout dropout;
    private final RotaryEmbedding rotary;
    public MultiHeadAttention(ExecutionContext context, int features, int heads, int kvHeads, long seed) {
        this(context, new Config(features, heads, kvHeads, false, true, 10000, 0), seed);
    }

    public MultiHeadAttention(ExecutionContext context, Config config, long seed) {
        this(context, config, seed, ParameterInitialization.DEFAULT);
    }

    public MultiHeadAttention(ExecutionContext context, Config config, long seed, ParameterInitialization initialization) {
        super(context);
        this.config = config;
        int kvFeatures = Math.multiplyExact(config.kvHeads, config.features / config.heads);
        try {
            query = register("query", new Linear(context, config.features, config.features, seed, config.bias, initialization));
            key = register("key", new Linear(context, config.features, kvFeatures, seed + 1, config.bias, initialization));
            value = register("value", new Linear(context, config.features, kvFeatures, seed + 2, config.bias, initialization));
            output = register("output", new Linear(context, config.features, config.features, seed + 3, config.bias, initialization));
            dropout = register("dropout", new Dropout(context, config.dropout, seed + 4));
            rotary = config.rotary ? register("rotary", new RotaryEmbedding(context, config.rotaryBase, true)) : null;
        } catch (RuntimeException | Error error) { close(); throw error; }
    }

    @Override
    public Tensor forward(Tensor input) {
        return forward(input, null, true);
    }

    public Tensor forward(Tensor input, Tensor allowed, boolean causal) {
        return compute(input, allowed, causal, null);
    }

    public Tensor decode(Tensor input, KvCache cache, Tensor allowed) {
        ensureOpen();
        if (context.activeTape() != null) {
            throw new IllegalStateException("Cached attention is inference-only");
        }
        return compute(input, allowed, true, cache);
    }

    private Tensor compute(Tensor input, Tensor allowed, boolean causal, KvCache cache) {
        ensureOpen();
        context.requireContext(input);
        if (input.shape().rank() != 3 || input.shape().dimension(2) != config.features || input.shape().dimension(1) == 0) {
            throw new IllegalArgumentException("Attention input must be nonempty [batch,time,features]");
        }
        long batch = input.shape().dimension(0), time = input.shape().dimension(1);
        long position = cache == null ? 0 : cache.length();
        try (Tensor q = project(query, input, batch, time, config.heads);
             Tensor k = project(key, input, batch, time, config.kvHeads);
             Tensor v = project(value, input, batch, time, config.kvHeads);
             Tensor rq = rotary == null ? q.retain() : rotary.forward(q, position);
             Tensor rk = rotary == null ? k.retain() : rotary.forward(k, position)) {
            if (cache != null) {
                cache.append(rk, v);
            }
            try (Tensor keys = cache == null ? rk.retain() : cache.keys(); Tensor values = cache == null ? v.retain() : cache.values();
                 Tensor attended = Attention.scaledDotProduct(rq, keys, values, allowed, causal, position, dropout);
                 Tensor ordered = Ops.permute(attended, 0, 2, 1, 3); Tensor packed = Ops.contiguous(ordered);
                 Tensor merged = Ops.reshape(packed, Shape.of(batch, time, config.features))) {
                return output.forward(merged);
            }
        }
    }

    private Tensor project(Linear projection, Tensor input, long batch, long time, int heads) {
        try (Tensor values = projection.forward(input); Tensor split = Ops.reshape(values, Shape.of(batch, time, heads, config.features / config.heads))) {
            return Ops.permute(split, 0, 2, 1, 3);
        }
    }

    @Override
    protected java.util.Map<String, String> localConfiguration() {
        return java.util.Map.of("features", Integer.toString(config.features), "heads", Integer.toString(config.heads),
                "kvHeads", Integer.toString(config.kvHeads));
    }

    public record Config(int features, int heads, int kvHeads, boolean bias, boolean rotary, double rotaryBase,
                         float dropout) {
        public Config {
            if (features <= 0 || heads <= 0 || kvHeads <= 0 || features % heads != 0 || heads % kvHeads != 0
                    || rotary && features / heads % 2 != 0 || !Double.isFinite(rotaryBase) || rotaryBase <= 0
                    || !Float.isFinite(dropout) || dropout < 0 || dropout >= 1) {
                throw new IllegalArgumentException("Invalid attention configuration");
            }
        }
    }
}

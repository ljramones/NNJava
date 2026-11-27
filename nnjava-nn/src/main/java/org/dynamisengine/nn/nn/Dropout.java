package org.dynamisengine.nn.nn;

import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Tensor;

import java.util.Map;

/**
 * Inverted dropout with checkpointable counter-based random streams.
 */
public final class Dropout extends AbstractModule implements UnaryModule {
    private final float probability;
    private long seed;
    private long counter;

    public Dropout(ExecutionContext context, float probability, long seed) {
        super(context);
        if (!Float.isFinite(probability) || probability < 0 || probability >= 1) {
            throw new IllegalArgumentException("Invalid dropout probability");
        }
        this.probability = probability;
        this.seed = seed;
    }

    @Override
    public Tensor forward(Tensor input) {
        ensureOpen();
        context.requireContext(input);
        if (!isTraining() || probability == 0) {
            return input.retain();
        }
        long next = Math.addExact(counter, input.shape().elements());
        Tensor result = Ops.dropout(input, probability, seed, counter);
        counter = next;
        return result;
    }

    @Override
    protected Map<String, Long> localState() {
        return Map.of("seed", seed, "counter", counter);
    }

    @Override
    protected Map<String, String> localConfiguration() {
        return Map.of("probability", Float.toHexString(probability));
    }

    @Override
    protected void validateLocalState(Map<String, Long> state) {
        if (state.get("counter") < 0) {
            throw new IllegalArgumentException("Negative dropout counter");
        }
    }

    @Override
    protected void restoreLocalState(Map<String, Long> state) {
        seed = state.get("seed");
        counter = state.get("counter");
    }
}

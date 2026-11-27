package org.dynamisengine.nn.training;

import org.dynamisengine.nn.backend.OptimizerCall;
import org.dynamisengine.nn.nn.Parameter;

import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * Decoupled AdamW with per-parameter bias-correction counters and restorable moments.
 */
public final class AdamW extends BaseOptimizer {
    private final float beta1, beta2, epsilon;

    public AdamW(Collection<Parameter> parameters, float learningRate) {
        this(List.of(ParameterGroup.of(parameters, learningRate, 0)), .9f, .999f, 1e-8f, LearningRateSchedule.constant());
    }

    public AdamW(Map<String, Parameter> parameters, float learningRate) {
        this(List.of(new ParameterGroup(parameters, learningRate, 0)), .9f, .999f, 1e-8f, LearningRateSchedule.constant());
    }

    public AdamW(List<ParameterGroup> groups, float beta1, float beta2, float epsilon, LearningRateSchedule schedule) {
        super("adamw", groups, schedule, options(beta1, beta2, epsilon), List.of("first", "second"));
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    private static Map<String, String> options(float b1, float b2, float epsilon) {
        new OptimizerCall.Settings(OptimizerCall.Kind.ADAMW, 0, 0, b1, b2, epsilon, false, 1);
        return Map.of("beta1", Float.toHexString(b1), "beta2", Float.toHexString(b2), "epsilon", Float.toHexString(epsilon));
    }

    @Override
    protected OptimizerCall.Settings settings(float rate, float decay, long step) {
        return new OptimizerCall.Settings(OptimizerCall.Kind.ADAMW, rate, decay, beta1, beta2, epsilon, false, step);
    }
}

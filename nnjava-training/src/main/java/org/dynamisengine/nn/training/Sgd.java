package org.dynamisengine.nn.training;

import org.dynamisengine.nn.backend.OptimizerCall;
import org.dynamisengine.nn.nn.Parameter;

import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * Vectrix SGD with optional momentum, Nesterov acceleration, and coupled decay.
 */
public final class Sgd extends BaseOptimizer {
    private final float momentum;
    private final boolean nesterov;

    public Sgd(Collection<Parameter> parameters, float learningRate) {
        this(parameters, learningRate, 0, 0, false);
    }

    public Sgd(Map<String, Parameter> parameters, float learningRate) {
        this(List.of(new ParameterGroup(parameters, learningRate, 0)), 0, false, LearningRateSchedule.constant());
    }

    public Sgd(Collection<Parameter> parameters, float learningRate, float momentum, float decay, boolean nesterov) {
        this(List.of(ParameterGroup.of(parameters, learningRate, decay)), momentum, nesterov, LearningRateSchedule.constant());
    }

    public Sgd(List<ParameterGroup> groups, float momentum, boolean nesterov, LearningRateSchedule schedule) {
        super("sgd", groups, schedule, options(momentum, nesterov), List.of("momentum"));
        this.momentum = momentum;
        this.nesterov = nesterov;
    }

    private static Map<String, String> options(float momentum, boolean nesterov) {
        new OptimizerCall.Settings(OptimizerCall.Kind.SGD, 0, 0, momentum, 0, 0, nesterov, 1);
        return Map.of("momentum", Float.toHexString(momentum), "nesterov", Boolean.toString(nesterov));
    }

    @Override
    protected OptimizerCall.Settings settings(float rate, float decay, long step) {
        return new OptimizerCall.Settings(OptimizerCall.Kind.SGD, rate, decay, momentum, 0, 0, nesterov, step);
    }
}

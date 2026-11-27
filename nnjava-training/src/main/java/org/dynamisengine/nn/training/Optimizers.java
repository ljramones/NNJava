package org.dynamisengine.nn.training;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.dynamisengine.nn.backend.OptimizerCall;
import org.dynamisengine.nn.nn.Parameter;

/** Reusable optimizer definitions; slots are allocated only when bound to parameters. */
public final class Optimizers {
    private Optimizers() { }

    public static AdamWBuilder adamW() { return new AdamWBuilder(); }
    public static SgdBuilder sgd() { return new SgdBuilder(); }

    private static void validate(float rate, float decay, LearningRateSchedule schedule) {
        if (!Float.isFinite(rate) || rate < 0 || !Float.isFinite(decay) || decay < 0) {
            throw new IllegalArgumentException("Nonnegative finite learning rate and decay required");
        }
        Objects.requireNonNull(schedule);
    }

    public record AdamWDefinition(float learningRate, float weightDecay, float beta1, float beta2,
                                  float epsilon, LearningRateSchedule schedule) implements OptimizerFactory {
        public AdamWDefinition {
            validate(learningRate, weightDecay, schedule);
            new OptimizerCall.Settings(OptimizerCall.Kind.ADAMW, learningRate, weightDecay, beta1, beta2, epsilon, false, 1);
        }

        @Override public Optimizer build(Map<String, Parameter> parameters) {
            return new AdamW(List.of(new ParameterGroup(parameters, learningRate, weightDecay)), beta1, beta2, epsilon, schedule);
        }
    }

    public record SgdDefinition(float learningRate, float weightDecay, float momentum, boolean nesterov,
                                LearningRateSchedule schedule) implements OptimizerFactory {
        public SgdDefinition {
            validate(learningRate, weightDecay, schedule);
            new OptimizerCall.Settings(OptimizerCall.Kind.SGD, learningRate, weightDecay, momentum, 0, 0, nesterov, 1);
        }

        @Override public Optimizer build(Map<String, Parameter> parameters) {
            return new Sgd(List.of(new ParameterGroup(parameters, learningRate, weightDecay)), momentum, nesterov, schedule);
        }
    }

    public static final class AdamWBuilder {
        private float learningRate = .001f, weightDecay, beta1 = .9f, beta2 = .999f, epsilon = 1e-8f;
        private LearningRateSchedule schedule = LearningRateSchedule.constant();
        public AdamWBuilder learningRate(float value) { learningRate = value; return this; }
        public AdamWBuilder weightDecay(float value) { weightDecay = value; return this; }
        public AdamWBuilder betas(float first, float second) { beta1 = first; beta2 = second; return this; }
        public AdamWBuilder epsilon(float value) { epsilon = value; return this; }
        public AdamWBuilder schedule(LearningRateSchedule value) { schedule = value; return this; }
        public AdamWDefinition define() { return new AdamWDefinition(learningRate, weightDecay, beta1, beta2, epsilon, schedule); }
    }

    public static final class SgdBuilder {
        private float learningRate = .01f, weightDecay, momentum;
        private boolean nesterov;
        private LearningRateSchedule schedule = LearningRateSchedule.constant();
        public SgdBuilder learningRate(float value) { learningRate = value; return this; }
        public SgdBuilder weightDecay(float value) { weightDecay = value; return this; }
        public SgdBuilder momentum(float value) { momentum = value; return this; }
        public SgdBuilder nesterov(boolean value) { nesterov = value; return this; }
        public SgdBuilder schedule(LearningRateSchedule value) { schedule = value; return this; }
        public SgdDefinition define() { return new SgdDefinition(learningRate, weightDecay, momentum, nesterov, schedule); }
    }
}

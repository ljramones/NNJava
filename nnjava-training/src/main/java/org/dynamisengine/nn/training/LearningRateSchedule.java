package org.dynamisengine.nn.training;

/**
 * Schedules receive a one-based optimizer step and a group's base rate.
 */
public interface LearningRateSchedule {
    static LearningRateSchedule constant() {
        return new Constant();
    }

    static LearningRateSchedule warmupCosine(long warmupSteps, long totalSteps, float minimumFactor) {
        return new WarmupCosine(warmupSteps, totalSteps, minimumFactor);
    }

    static LearningRateSchedule stepDecay(long interval, float factor) {
        return new StepDecay(interval, factor);
    }

    float rate(long step, float baseRate);

    String id();

    record Constant() implements LearningRateSchedule {
        public float rate(long step, float baseRate) {
            return baseRate;
        }

        public String id() {
            return "constant";
        }
    }

    record WarmupCosine(long warmupSteps, long totalSteps, float minimumFactor) implements LearningRateSchedule {
        public WarmupCosine {
            if (warmupSteps < 0 || totalSteps <= warmupSteps || !Float.isFinite(minimumFactor) || minimumFactor < 0 || minimumFactor > 1) {
                throw new IllegalArgumentException("Invalid warmup/cosine schedule");
            }
        }

        public float rate(long step, float baseRate) {
            if (step < 1) {
                throw new IllegalArgumentException("Steps are one-based");
            }
            if (step <= warmupSteps) {
                return baseRate * ((float) step / warmupSteps);
            }
            double progress = Math.min(1.0, (double) (step - warmupSteps) / (totalSteps - warmupSteps));
            return (float) (baseRate * (minimumFactor + (1 - minimumFactor) * .5 * (1 + StrictMath.cos(Math.PI * progress))));
        }

        public String id() {
            return "warmupCosine:" + warmupSteps + ":" + totalSteps + ":" + minimumFactor;
        }
    }

    record StepDecay(long interval, float factor) implements LearningRateSchedule {
        public StepDecay {
            if (interval < 1 || !Float.isFinite(factor) || factor < 0 || factor > 1) {
                throw new IllegalArgumentException("Invalid step decay");
            }
        }

        public float rate(long step, float baseRate) {
            return (float) (baseRate * StrictMath.pow(factor, (step - 1) / interval));
        }

        public String id() {
            return "stepDecay:" + interval + ":" + factor;
        }
    }
}

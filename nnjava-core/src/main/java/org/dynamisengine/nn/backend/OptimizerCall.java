package org.dynamisengine.nn.backend;

import org.dynamisengine.nn.tensor.DType;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Explicit mutation of a parameter and optimizer slots; gradients are read-only.
 */
public record OptimizerCall(Settings settings, KernelTensor parameter, KernelTensor gradient,
                            List<KernelTensor> slots) {
    public OptimizerCall {
        slots = List.copyOf(slots);
        if (slots.size() != (settings.kind() == Kind.ADAMW ? 2 : 1)) {
            throw new IllegalArgumentException("Optimizer slot count");
        }
        Set<Buffer> allocations = new HashSet<>();
        for (KernelTensor tensor : java.util.stream.Stream.concat(java.util.stream.Stream.of(parameter, gradient), slots.stream()).toList()) {
            if (tensor.buffer().dtype() != DType.FP32 || !tensor.layout().shape().equals(parameter.layout().shape())
                    || !tensor.buffer().device().equals(parameter.buffer().device()) || !allocations.add(tensor.buffer())) {
                throw new IllegalArgumentException("Optimizer operands must be compatible and nonaliasing");
            }
        }
        if (!parameter.layout().isContiguous() || slots.stream().anyMatch(s -> !s.layout().isContiguous())) {
            throw new IllegalArgumentException("Optimizer state must be contiguous");
        }
    }

    public enum Kind {SGD, ADAMW}

    public record Settings(Kind kind, float learningRate, float weightDecay, float beta1, float beta2,
                           float epsilon, boolean nesterov, long step) {
        public Settings {
            if (kind == null || !Float.isFinite(learningRate) || learningRate < 0 || !Float.isFinite(weightDecay) || weightDecay < 0
                    || !Float.isFinite(beta1) || beta1 < 0 || beta1 >= 1 || step < 1) {
                throw new IllegalArgumentException("Invalid optimizer settings");
            }
            if (kind == Kind.ADAMW && (!Float.isFinite(beta2) || beta2 < 0 || beta2 >= 1 || !Float.isFinite(epsilon) || epsilon <= 0)) {
                throw new IllegalArgumentException("Invalid AdamW settings");
            }
            if (kind == Kind.SGD && nesterov && beta1 == 0) {
                throw new IllegalArgumentException("Nesterov requires momentum");
            }
        }
    }
}

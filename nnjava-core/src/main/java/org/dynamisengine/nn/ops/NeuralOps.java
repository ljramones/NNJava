package org.dynamisengine.nn.ops;

import org.dynamisengine.nn.backend.KernelAttributes;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;

import java.util.List;

/**
 * Neural operations with matching Vectrix forward and backward kernels.
 */
public final class NeuralOps {
    private NeuralOps() {
    }

    static Tensor activation(Tensor x, Operation op) {
        Operation backward = Operation.valueOf(op.name() + "_BACKWARD");
        return Ops.execute(op, List.of(x), x.shape(), 0, (dy, saved) -> new Tensor[]{
                Ops.execute(backward, List.of(saved.getFirst(), dy), saved.getFirst().shape(), 0, null)
        });
    }

    static Tensor softmax(Tensor x, int axis, boolean logarithmic) {
        int normalized = x.shape().axis(axis);
        if (normalized != x.shape().rank() - 1) {
            try (Tensor moved = Ops.transpose(x, normalized, x.shape().rank() - 1);
                 Tensor computed = softmax(moved, -1, logarithmic)) {
                return Ops.transpose(computed, normalized, x.shape().rank() - 1);
            }
        }
        Operation forward = logarithmic ? Operation.LOG_SOFTMAX : Operation.SOFTMAX;
        Operation backward = logarithmic ? Operation.LOG_SOFTMAX_BACKWARD : Operation.SOFTMAX_BACKWARD;
        return Ops.execute(forward, List.of(x), x.shape(), 0, (dy, saved) -> new Tensor[]{
                Ops.execute(backward, List.of(saved.getFirst(), dy), saved.getFirst().shape(), 0, null)
        });
    }

    static Tensor dropout(Tensor x, float p, long seed, long counter) {
        KernelAttributes attrs = KernelAttributes.of("p", p).and("seed", seed).and("counter", counter);
        return Ops.execute(Operation.DROPOUT, List.of(x), x.shape(), DType.FP32, 0, attrs,
                (dy, saved) -> new Tensor[]{dropout(dy, p, seed, counter)});
    }

    static Tensor norm(Tensor x, Tensor scale, Tensor bias, float epsilon) {
        boolean rms = bias == null;
        List<Tensor> inputs = rms ? List.of(x, scale) : List.of(x, scale, bias);
        KernelAttributes attrs = KernelAttributes.of("rms", rms).and("epsilon", epsilon);
        return Ops.execute(rms ? Operation.RMS_NORM : Operation.LAYER_NORM, inputs, x.shape(), DType.FP32, 0, attrs,
                (dy, saved) -> {
                    Tensor[] result = new Tensor[saved.size()];
                    var arguments = new java.util.ArrayList<>(saved);
                    arguments.add(dy);
                    try {
                        for (int i = 0; i < result.length; i++) {
                            if (saved.get(i).requiresGrad()) {
                                result[i] = Ops.execute(Operation.NORM_BACKWARD, arguments, saved.get(i).shape(),
                                        DType.FP32, 0, attrs.and("which", i), null);
                            }
                        }
                        return result;
                    } catch (RuntimeException error) {
                        for (Tensor value : result) {
                            if (value != null) {
                                value.close();
                            }
                        }
                        throw error;
                    }
                });
    }

    public static Tensor crossEntropy(Tensor logits, Tensor targets, String reduction, int ignoreIndex) {
        if (logits.shape().rank() < 1) {
            throw new IllegalArgumentException("Logits need a class axis");
        }
        Shape target = reduction.equals("none") ? targets.shape() : Shape.of();
        KernelAttributes attrs = KernelAttributes.of("reduction", reduction).and("ignore", ignoreIndex);
        return Ops.execute(Operation.CROSS_ENTROPY, List.of(logits, targets), target, DType.FP32, 0, attrs,
                (dy, saved) -> new Tensor[]{Ops.execute(Operation.CROSS_ENTROPY_BACKWARD,
                        List.of(saved.get(0), saved.get(1), dy), saved.getFirst().shape(), DType.FP32, 0, attrs, null), null});
    }

    static Tensor rope(Tensor x, Tensor positions, long offset, double base, boolean half, boolean inverse) {
        KernelAttributes attrs = KernelAttributes.of("offset", offset).and("base", base).and("half", half);
        List<Tensor> inputs = positions == null ? List.of(x) : List.of(x, positions);
        if (positions != null) {
            attrs = attrs.and("positions", true);
        }
        return Ops.execute(inverse ? Operation.ROPE_BACKWARD : Operation.ROPE, inputs, x.shape(), DType.FP32, 0, attrs,
                (dy, saved) -> {
                    Tensor derivative = rope(dy, saved.size() == 2 ? saved.get(1) : null, offset, base, half, !inverse);
                    return saved.size() == 1 ? new Tensor[]{derivative} : new Tensor[]{derivative, null};
                });
    }
}

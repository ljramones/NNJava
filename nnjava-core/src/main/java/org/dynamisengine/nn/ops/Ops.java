package org.dynamisengine.nn.ops;

import org.dynamisengine.nn.autograd.BackwardRule;
import org.dynamisengine.nn.autograd.GradientTape;
import org.dynamisengine.nn.backend.KernelAttributes;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.*;

import java.util.List;

/**
 * Functional tensor operations. Each returned tensor is an owned handle.
 */
public final class Ops {
    private Ops() {
    }

    public static Tensor add(Tensor left, Tensor right) {
        return binary(Operation.ADD, left, right, (gradient, inputs) -> new Tensor[]{
                gradientTo(gradient, inputs.get(0)), gradientTo(gradient, inputs.get(1))
        });
    }

    public static Tensor subtract(Tensor left, Tensor right) {
        return binary(Operation.SUBTRACT, left, right, (gradient, inputs) -> {
            Tensor rightGradient = null;
            if (inputs.get(1).requiresGrad()) {
                try (Tensor negative = scale(gradient, -1f)) {
                    rightGradient = sumTo(negative, inputs.get(1).shape());
                }
            }
            return new Tensor[]{gradientTo(gradient, inputs.get(0)), rightGradient};
        });
    }

    public static Tensor multiply(Tensor left, Tensor right) {
        return binary(Operation.MULTIPLY, left, right, (gradient, inputs) -> {
            Tensor first = null;
            Tensor second = null;
            try {
                if (inputs.get(0).requiresGrad()) {
                    try (Tensor product = multiply(gradient, inputs.get(1))) {
                        first = sumTo(product, inputs.get(0).shape());
                    }
                }
                if (inputs.get(1).requiresGrad()) {
                    try (Tensor product = multiply(gradient, inputs.get(0))) {
                        second = sumTo(product, inputs.get(1).shape());
                    }
                }
                return new Tensor[]{first, second};
            } catch (RuntimeException error) {
                if (first != null) {
                    first.close();
                }
                if (second != null) {
                    second.close();
                }
                throw error;
            }
        });
    }

    private static Tensor binary(Operation operation, Tensor left, Tensor right, BackwardRule backward) {
        requireFloat(left);
        requireFloat(right);
        left.context().requireContext(right);
        return execute(operation, List.of(left, right), Shape.broadcast(left.shape(), right.shape()), 0f, backward);
    }

    public static Tensor scale(Tensor input, float value) {
        requireFloat(input);
        return execute(Operation.SCALE, List.of(input), input.shape(), value,
                (gradient, inputs) -> new Tensor[]{scale(gradient, value)});
    }

    public static Tensor square(Tensor input) {
        return multiply(input, input);
    }

    public static Tensor sum(Tensor input) {
        requireFloat(input);
        return execute(Operation.SUM, List.of(input), Shape.of(), 0f,
                (gradient, inputs) -> new Tensor[]{broadcastTo(gradient, inputs.getFirst().shape())});
    }

    public static Tensor mean(Tensor input) {
        if (input.shape().elements() == 0) {
            throw new IllegalArgumentException("Mean over an empty domain is undefined");
        }
        try (Tensor total = sum(input)) {
            return scale(total, 1f / input.shape().elements());
        }
    }

    /**
     * Reduce a broadcast result back to an operand's original shape.
     */
    public static Tensor sumTo(Tensor input, Shape target) {
        requireFloat(input);
        if (!Shape.broadcast(target, input.shape()).equals(input.shape())) {
            throw new IllegalArgumentException("Invalid broadcast reduction from " + input.shape() + " to " + target);
        }
        return execute(Operation.SUM_TO, List.of(input), target, 0f,
                (gradient, inputs) -> new Tensor[]{broadcastTo(gradient, inputs.getFirst().shape())});
    }

    public static Tensor matmul(Tensor left, Tensor right) {
        requireFloat(left);
        requireFloat(right);
        left.context().requireContext(right);
        Shape output = Shape.matmul(left.shape(), right.shape());
        return execute(Operation.MATMUL, List.of(left, right), output, 0f, (gradient, inputs) -> {
            Tensor first = null;
            Tensor second = null;
            try {
                if (inputs.get(0).requiresGrad()) {
                    try (Tensor transposed = transpose(inputs.get(1))) {
                        try (Tensor product = matmul(gradient, transposed)) {
                            first = sumTo(product, inputs.get(0).shape());
                        }
                    }
                }
                if (inputs.get(1).requiresGrad()) {
                    try (Tensor transposed = transpose(inputs.get(0))) {
                        try (Tensor product = matmul(transposed, gradient)) {
                            second = sumTo(product, inputs.get(1).shape());
                        }
                    }
                }
                return new Tensor[]{first, second};
            } catch (RuntimeException error) {
                if (first != null) {
                    first.close();
                }
                if (second != null) {
                    second.close();
                }
                throw error;
            }
        });
    }

    public static Tensor copy(Tensor input) {
        return execute(Operation.COPY, List.of(input), input.shape(), 0f,
                (gradient, inputs) -> new Tensor[]{copy(gradient)});
    }

    public static Tensor contiguous(Tensor input) {
        return input.layout().isContiguous() ? input.retain() : copy(input);
    }

    public static Tensor reshape(Tensor input, Shape target) {
        return view(input, input.layout().reshape(target), false, (gradient, inputs) -> {
            try (Tensor packed = contiguous(gradient)) {
                return new Tensor[]{reshape(packed, inputs.getFirst().shape())};
            }
        });
    }

    public static Tensor transpose(Tensor input) {
        if (input.shape().rank() < 2) {
            throw new IllegalArgumentException("Transpose needs at least two axes");
        }
        return transpose(input, input.shape().rank() - 2, input.shape().rank() - 1);
    }

    public static Tensor transpose(Tensor input, int first, int second) {
        int a = input.shape().axis(first), b = input.shape().axis(second);
        return view(input, input.layout().transpose(a, b), false,
                (gradient, inputs) -> new Tensor[]{transpose(gradient, a, b)});
    }

    public static Tensor broadcastTo(Tensor input, Shape target) {
        return view(input, input.layout().broadcastTo(target), true,
                (gradient, inputs) -> new Tensor[]{sumTo(gradient, inputs.getFirst().shape())});
    }

    static Tensor view(Tensor input, TensorLayout layout, boolean readOnly, BackwardRule backward) {
        ExecutionContext context = input.context();
        GradientTape tape = context.activeTape();
        boolean record = tape != null && tape.watches(List.of(input));
        Tensor output = TensorAccess.view(input, layout, readOnly);
        try {
            if (record) {
                tape.record(output, List.of(input), backward);
            }
            return output;
        } catch (RuntimeException error) {
            output.close();
            throw error;
        }
    }

    static Tensor execute(Operation operation, List<Tensor> inputs, Shape shape,
                          float scalar, BackwardRule backward) {
        return execute(operation, inputs, shape, inputs.getFirst().dtype(), scalar, KernelAttributes.EMPTY, backward);
    }

    static Tensor execute(Operation operation, List<Tensor> inputs, Shape shape, DType dtype,
                          float scalar, KernelAttributes attributes, BackwardRule backward) {
        ExecutionContext context = inputs.getFirst().context();
        for (Tensor input : inputs) {
            context.requireContext(input);
        }
        GradientTape tape = context.activeTape();
        boolean record = backward != null && dtype == DType.FP32 && tape != null && tape.watches(inputs);
        Tensor output = context.empty(dtype, shape);
        try {
            context.submit(operation, inputs, output, scalar, attributes);
            if (record) {
                tape.record(output, inputs, backward);
            }
            return output;
        } catch (RuntimeException error) {
            output.close();
            throw error;
        }
    }

    public static Tensor divide(Tensor a, Tensor b) {
        return MoreOps.divide(a, b);
    }

    /** Channel-first cross-correlation with weights [outputs,inputs/groups,kernelHeight,kernelWidth]. */
    public static Tensor conv2d(Tensor input, Tensor weights, Tensor bias, Conv2dConfig config) {
        return SpatialOps.conv2d(input,weights,bias,config);
    }

    public static Tensor conv2d(Tensor input, Tensor weights, Conv2dConfig config) { return conv2d(input,weights,null,config); }

    /** Padding never wins; equal maxima route gradients to the first row-major value. */
    public static Tensor maxPool2d(Tensor input, Pool2dConfig config) { return SpatialOps.pool2d(input,config,false); }

    /** Averages only in-bounds cells; padding is excluded from the denominator. */
    public static Tensor avgPool2d(Tensor input, Pool2dConfig config) { return SpatialOps.pool2d(input,config,true); }

    public static Tensor exp(Tensor x) {
        return MoreOps.unary(x, Operation.EXP);
    }

    public static Tensor log(Tensor x) {
        return MoreOps.unary(x, Operation.LOG);
    }

    public static Tensor sqrt(Tensor x) {
        return MoreOps.unary(x, Operation.SQRT);
    }

    public static Tensor sum(Tensor x, int axis, boolean keep) {
        return sum(x, new int[]{axis}, keep);
    }

    public static Tensor sum(Tensor x, int[] axes, boolean keep) {
        return MoreOps.reduce(x, axes, keep, Operation.REDUCE_SUM);
    }

    public static Tensor mean(Tensor x, int axis, boolean keep) {
        return mean(x, new int[]{axis}, keep);
    }

    public static Tensor mean(Tensor x, int[] axes, boolean keep) {
        return MoreOps.mean(x, axes, keep);
    }

    public static Tensor max(Tensor x, int axis, boolean keep) {
        return MoreOps.reduce(x, new int[]{axis}, keep, Operation.REDUCE_MAX);
    }

    public static Tensor argmax(Tensor x, int axis, boolean keep) {
        return MoreOps.reduce(x, new int[]{axis}, keep, Operation.ARGMAX);
    }

    public static Tensor permute(Tensor x, int... axes) {
        return MoreOps.permute(x, axes);
    }

    public static Tensor slice(Tensor x, int axis, long start, long end) {
        return slice(x, axis, start, end, 1);
    }

    public static Tensor slice(Tensor x, int axis, long start, long end, long step) {
        return MoreOps.slice(x, axis, start, end, step);
    }

    public static Tensor gather(Tensor x, int axis, Tensor indices) {
        return MoreOps.gather(x, axis, indices);
    }

    public static Tensor scatterAdd(Tensor values, int axis, Tensor indices, Shape target) {
        return MoreOps.scatter(values, axis, indices, target);
    }

    public static Tensor concat(int axis, Tensor... inputs) {
        return MoreOps.concat(axis, List.of(inputs));
    }

    /**
     * Return owned views whose lengths exactly partition an axis.
     */
    public static List<Tensor> split(Tensor input, int axis, long... sizes) {
        int normalized = input.shape().axis(axis);
        long total = 0;
        for (long size : sizes) {
            if (size < 0) {
                throw new IllegalArgumentException("Negative split size");
            }
            total = Math.addExact(total, size);
        }
        if (total != input.shape().dimension(normalized)) {
            throw new IllegalArgumentException("Split sizes must cover the axis");
        }
        java.util.ArrayList<Tensor> result = new java.util.ArrayList<>();
        long start = 0;
        try {
            for (long size : sizes) {
                result.add(slice(input, normalized, start, start + size));
                start += size;
            }
            return List.copyOf(result);
        } catch (RuntimeException error) {
            result.forEach(Tensor::close);
            throw error;
        }
    }

    public static Tensor cast(Tensor x, DType dtype) {
        return MoreOps.cast(x, dtype);
    }

    public static Tensor equal(Tensor a, Tensor b) {
        return MoreOps.condition(a, b, Operation.EQUAL);
    }

    public static Tensor less(Tensor a, Tensor b) {
        return MoreOps.condition(a, b, Operation.LESS);
    }

    public static Tensor greater(Tensor a, Tensor b) {
        return MoreOps.condition(a, b, Operation.GREATER);
    }

    public static Tensor and(Tensor a, Tensor b) {
        return MoreOps.condition(a, b, Operation.AND);
    }

    public static Tensor or(Tensor a, Tensor b) {
        return MoreOps.condition(a, b, Operation.OR);
    }

    public static Tensor not(Tensor x) {
        return execute(Operation.NOT, List.of(x), x.shape(), DType.BOOL, 0, KernelAttributes.EMPTY, null);
    }

    public static Tensor where(Tensor condition, Tensor a, Tensor b) {
        return MoreOps.where(condition, a, b);
    }

    public static Tensor isFinite(Tensor x) {
        return execute(Operation.IS_FINITE, List.of(x), x.shape(), DType.BOOL, 0, KernelAttributes.EMPTY, null);
    }

    public static Tensor all(Tensor x) {
        return execute(Operation.ALL, List.of(x), Shape.of(), DType.BOOL, 0, KernelAttributes.EMPTY, null);
    }

    /**
     * Stable vector norm for diagnostics and gradient clipping; not recorded for differentiation.
     */
    public static Tensor norm(Tensor x) {
        return execute(Operation.NORM2, List.of(x), Shape.of(), 0, null);
    }

    public static Tensor relu(Tensor x) {
        return NeuralOps.activation(x, Operation.RELU);
    }

    public static Tensor sigmoid(Tensor x) {
        return NeuralOps.activation(x, Operation.SIGMOID);
    }

    public static Tensor tanh(Tensor x) {
        return NeuralOps.activation(x, Operation.TANH);
    }

    public static Tensor gelu(Tensor x) {
        return NeuralOps.activation(x, Operation.GELU);
    }

    public static Tensor silu(Tensor x) {
        return NeuralOps.activation(x, Operation.SILU);
    }

    public static Tensor softmax(Tensor x, int axis) {
        return NeuralOps.softmax(x, axis, false);
    }

    public static Tensor logSoftmax(Tensor x, int axis) {
        return NeuralOps.softmax(x, axis, true);
    }

    public static Tensor dropout(Tensor x, float p, long seed, long counter) {
        return NeuralOps.dropout(x, p, seed, counter);
    }

    public static Tensor layerNorm(Tensor x, Tensor scale, Tensor bias, float epsilon) {
        return NeuralOps.norm(x, scale, bias, epsilon);
    }

    public static Tensor rmsNorm(Tensor x, Tensor scale, float epsilon) {
        return NeuralOps.norm(x, scale, null, epsilon);
    }

    public static Tensor rope(Tensor x, long offset, double base, boolean half) {
        return NeuralOps.rope(x, null, offset, base, half, false);
    }

    public static Tensor rope(Tensor x, Tensor positions, double base, boolean half) {
        return NeuralOps.rope(x, positions, 0, base, half, false);
    }

    private static Tensor gradientTo(Tensor gradient, Tensor input) {
        return input.requiresGrad() ? sumTo(gradient, input.shape()) : null;
    }

    private static void requireFloat(Tensor tensor) {
        if (tensor.dtype() != DType.FP32) {
            throw new IllegalArgumentException("Arithmetic requires FP32; got " + tensor.dtype());
        }
    }
}

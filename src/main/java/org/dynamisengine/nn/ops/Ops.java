package org.dynamisengine.nn.ops;

import java.util.List;
import org.dynamisengine.nn.autograd.BackwardRule;
import org.dynamisengine.nn.autograd.GradientTape;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;
import org.dynamisengine.nn.tensor.TensorAccess;
import org.dynamisengine.nn.tensor.TensorLayout;

/** Functional tensor operations. Each returned tensor is an owned handle. */
public final class Ops {
    private Ops() {
    }

    public static Tensor add(Tensor left, Tensor right) {
        return binary(Operation.ADD, left, right, (gradient, inputs) -> new Tensor[] {
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
            return new Tensor[] {gradientTo(gradient, inputs.get(0)), rightGradient};
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
                return new Tensor[] {first, second};
            } catch (RuntimeException error) {
                if (first != null) { first.close(); }
                if (second != null) { second.close(); }
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
                (gradient, inputs) -> new Tensor[] {scale(gradient, value)});
    }

    public static Tensor square(Tensor input) {
        return multiply(input, input);
    }

    public static Tensor sum(Tensor input) {
        requireFloat(input);
        return execute(Operation.SUM, List.of(input), Shape.of(), 0f,
                (gradient, inputs) -> new Tensor[] {broadcastTo(gradient, inputs.getFirst().shape())});
    }

    public static Tensor mean(Tensor input) {
        if (input.shape().elements() == 0) {
            throw new IllegalArgumentException("Mean over an empty domain is undefined");
        }
        try (Tensor total = sum(input)) {
            return scale(total, 1f / input.shape().elements());
        }
    }

    /** Reduce a broadcast result back to an operand's original shape. */
    public static Tensor sumTo(Tensor input, Shape target) {
        requireFloat(input);
        if (!Shape.broadcast(target, input.shape()).equals(input.shape())) {
            throw new IllegalArgumentException("Invalid broadcast reduction from " + input.shape() + " to " + target);
        }
        return execute(Operation.SUM_TO, List.of(input), target, 0f,
                (gradient, inputs) -> new Tensor[] {broadcastTo(gradient, inputs.getFirst().shape())});
    }

    public static Tensor matmul(Tensor left, Tensor right) {
        requireFloat(left);
        requireFloat(right);
        left.context().requireContext(right);
        if (left.shape().rank() != 2 || right.shape().rank() != 2
                || left.shape().dimension(1) != right.shape().dimension(0)) {
            throw new IllegalArgumentException("matmul requires compatible rank-two tensors");
        }
        Shape output = Shape.of(left.shape().dimension(0), right.shape().dimension(1));
        return execute(Operation.MATMUL, List.of(left, right), output, 0f, (gradient, inputs) -> {
            Tensor first = null;
            Tensor second = null;
            try {
                if (inputs.get(0).requiresGrad()) {
                    try (Tensor transposed = transpose(inputs.get(1))) {
                        first = matmul(gradient, transposed);
                    }
                }
                if (inputs.get(1).requiresGrad()) {
                    try (Tensor transposed = transpose(inputs.get(0))) {
                        second = matmul(transposed, gradient);
                    }
                }
                return new Tensor[] {first, second};
            } catch (RuntimeException error) {
                if (first != null) { first.close(); }
                if (second != null) { second.close(); }
                throw error;
            }
        });
    }

    public static Tensor copy(Tensor input) {
        return execute(Operation.COPY, List.of(input), input.shape(), 0f,
                (gradient, inputs) -> new Tensor[] {copy(gradient)});
    }

    public static Tensor contiguous(Tensor input) {
        return input.layout().isContiguous() ? input.retain() : copy(input);
    }

    public static Tensor reshape(Tensor input, Shape target) {
        return view(input, input.layout().reshape(target), false, (gradient, inputs) -> {
            try (Tensor packed = contiguous(gradient)) {
                return new Tensor[] {reshape(packed, inputs.getFirst().shape())};
            }
        });
    }

    public static Tensor transpose(Tensor input) {
        if (input.shape().rank() != 2) {
            throw new IllegalArgumentException("Use explicit axes to transpose a non-matrix tensor");
        }
        return transpose(input, 0, 1);
    }

    public static Tensor transpose(Tensor input, int first, int second) {
        return view(input, input.layout().transpose(first, second), false,
                (gradient, inputs) -> new Tensor[] {transpose(gradient, first, second)});
    }

    public static Tensor broadcastTo(Tensor input, Shape target) {
        return view(input, input.layout().broadcastTo(target), true,
                (gradient, inputs) -> new Tensor[] {sumTo(gradient, inputs.getFirst().shape())});
    }

    private static Tensor view(Tensor input, TensorLayout layout, boolean readOnly, BackwardRule backward) {
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

    private static Tensor execute(Operation operation, List<Tensor> inputs, Shape shape,
            float scalar, BackwardRule backward) {
        ExecutionContext context = inputs.getFirst().context();
        for (Tensor input : inputs) {
            context.requireContext(input);
        }
        GradientTape tape = context.activeTape();
        boolean record = tape != null && tape.watches(inputs);
        Tensor output = context.empty(inputs.getFirst().dtype(), shape);
        try {
            context.submit(operation, inputs, output, scalar);
            if (record) {
                tape.record(output, inputs, backward);
            }
            return output;
        } catch (RuntimeException error) {
            output.close();
            throw error;
        }
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

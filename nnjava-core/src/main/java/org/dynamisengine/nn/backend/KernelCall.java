package org.dynamisengine.nn.backend;

import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Validated functional dispatch; AXPY explicitly accumulates into its output.
 */
public record KernelCall(Operation operation, List<KernelTensor> inputs, KernelTensor output,
                         float scalar, KernelAttributes attributes) {
    public KernelCall(Operation operation, List<KernelTensor> inputs, KernelTensor output, float scalar) {
        this(operation, inputs, output, scalar, KernelAttributes.EMPTY);
    }

    public KernelCall {
        Objects.requireNonNull(operation, "operation");
        inputs = List.copyOf(inputs);
        Objects.requireNonNull(output, "output");
        Objects.requireNonNull(attributes, "attributes");
        int count = switch (operation) {
            case FILL -> 0;
            case ADD, SUBTRACT, MULTIPLY, DIVIDE, MATMUL, EQUAL, LESS, GREATER, AND, OR,
                 GATHER, SCATTER_ADD, MAX_BACKWARD, CROSS_ENTROPY,
                 RELU_BACKWARD, SIGMOID_BACKWARD, TANH_BACKWARD, GELU_BACKWARD, SILU_BACKWARD,
                 SOFTMAX_BACKWARD, LOG_SOFTMAX_BACKWARD -> 2;
            case WHERE, LAYER_NORM, CROSS_ENTROPY_BACKWARD -> 3;
            case RMS_NORM -> 2;
            case NORM_BACKWARD -> attributes.flag("rms") ? 3 : 4;
            case ROPE, ROPE_BACKWARD -> attributes.has("positions") ? 2 : 1;
            case CONCAT -> inputs.size();
            default -> 1;
        };
        if (inputs.size() != count || operation == Operation.CONCAT && inputs.isEmpty()) {
            throw new IllegalArgumentException(operation + " has invalid input count");
        }
        if (operation == Operation.NORM_BACKWARD) {
            int which = attributes.intValue("which");
            require(which >= 0 && which < (attributes.flag("rms") ? 2 : 3), "Normalization gradient index");
        }
        for (KernelTensor input : inputs) {
            if (!input.buffer().device().equals(output.buffer().device()) || input.buffer() == output.buffer()) {
                throw new IllegalArgumentException("Operands must share a device and outputs must not alias inputs");
            }
        }
        DType out = output.buffer().dtype();
        switch (operation) {
            case CAST -> {
            }
            case EQUAL, LESS, GREATER -> {
                require(out == DType.BOOL && inputs.get(0).buffer().dtype() == inputs.get(1).buffer().dtype(), "Comparison types");
            }
            case AND, OR, NOT, ALL -> {
                require(out == DType.BOOL && inputs.stream().allMatch(i -> i.buffer().dtype() == DType.BOOL), "Boolean types");
            }
            case IS_FINITE ->
                    require(out == DType.BOOL && inputs.getFirst().buffer().dtype() == DType.FP32, "Finite check types");
            case ARGMAX ->
                    require(out == DType.INT32 && inputs.getFirst().buffer().dtype() == DType.FP32, "Argmax types");
            case WHERE -> require(inputs.get(0).buffer().dtype() == DType.BOOL
                    && inputs.get(1).buffer().dtype() == out && inputs.get(2).buffer().dtype() == out, "Selection types");
            case GATHER, SCATTER_ADD -> require(inputs.get(0).buffer().dtype() == out
                    && inputs.get(1).buffer().dtype() == DType.INT32, "Indexing types");
            case CROSS_ENTROPY, CROSS_ENTROPY_BACKWARD -> require(out == DType.FP32
                    && inputs.get(0).buffer().dtype() == DType.FP32 && inputs.get(1).buffer().dtype() == DType.INT32
                    && (inputs.size() == 2 || inputs.get(2).buffer().dtype() == DType.FP32), "Loss types");
            case ROPE, ROPE_BACKWARD -> require(out == DType.FP32 && inputs.get(0).buffer().dtype() == DType.FP32
                    && (inputs.size() == 1 || inputs.get(1).buffer().dtype() == DType.INT32), "Position types");
            default -> {
                require(inputs.stream().allMatch(i -> i.buffer().dtype() == out), "Operand types differ");
                require(operation == Operation.COPY || operation == Operation.CONCAT || out == DType.FP32, "FP32 required");
            }
        }
        Shape actual = output.layout().shape();
        Shape first = inputs.isEmpty() ? actual : inputs.getFirst().layout().shape();
        Shape expected = switch (operation) {
            case FILL -> actual;
            case SUM, ALL, NORM2 -> Shape.of();
            case ADD, SUBTRACT, MULTIPLY, DIVIDE, EQUAL, LESS, GREATER, AND, OR ->
                    Shape.broadcast(first, inputs.get(1).layout().shape());
            case WHERE ->
                    Shape.broadcast(first, Shape.broadcast(inputs.get(1).layout().shape(), inputs.get(2).layout().shape()));
            case MATMUL -> Shape.matmul(first, inputs.get(1).layout().shape());
            case SUM_TO -> {
                require(Shape.broadcast(actual, first).equals(first), "Invalid broadcast reduction");
                yield actual;
            }
            case REDUCE_SUM, REDUCE_MAX, ARGMAX -> reducedShape(first, attributes.axes(), attributes.flag("keep"));
            case GATHER -> gatherShape(first, inputs.get(1).layout().shape(), attributes.intValue("axis"));
            case SCATTER_ADD -> {
                require(gatherShape(actual, inputs.get(1).layout().shape(), attributes.intValue("axis")).equals(first), "Scatter shape");
                yield actual;
            }
            case CONCAT ->
                    concatShape(inputs.stream().map(i -> i.layout().shape()).toList(), attributes.intValue("axis"));
            case SLICE_BACKWARD -> {
                Shape source = attributes.shape("shape");
                require(org.dynamisengine.nn.tensor.TensorLayout.contiguous(source).slice(attributes.intValue("axis"),
                        attributes.integer("start"), attributes.integer("end"), attributes.integer("step")).shape().equals(first), "Slice gradient shape");
                yield source;
            }
            case CROSS_ENTROPY, CROSS_ENTROPY_BACKWARD -> {
                require(first.rank() >= 1 && first.dimension(first.rank() - 1) > 0, "Loss class dimension");
                Shape rows = first.prefix(first.rank() - 1);
                require(inputs.get(1).layout().shape().equals(rows), "Loss target shape");
                String reduction = attributes.text("reduction");
                require(List.of("none", "sum", "mean").contains(reduction), "Loss reduction");
                Shape loss = reduction.equals("none") ? rows : Shape.of();
                if (operation == Operation.CROSS_ENTROPY_BACKWARD) {
                    require(inputs.get(2).layout().shape().equals(loss), "Loss seed shape");
                    yield first;
                }
                yield loss;
            }
            case NORM_BACKWARD -> inputs.get(attributes.intValue("which")).layout().shape();
            default -> first;
        };
        require(actual.equals(expected), "Output shape for " + operation + ": expected " + expected + ", got " + actual);
        if (operation == Operation.MAX_BACKWARD) {
            require(inputs.get(1).layout().shape().equals(reducedShape(first, attributes.axes(), attributes.flag("keep"))),
                    "Maximum backward seed shape");
        }
        require(operation == Operation.COPY || output.layout().isContiguous(), "Output must be contiguous");
        if (operation == Operation.REDUCE_MAX || operation == Operation.ARGMAX) {
            for (int axis : attributes.axes()) {
                require(first.dimension(first.axis(axis)) > 0, "Empty maximum reduction");
            }
        }
        if (List.of(Operation.RELU_BACKWARD, Operation.SIGMOID_BACKWARD, Operation.TANH_BACKWARD,
                Operation.GELU_BACKWARD, Operation.SILU_BACKWARD, Operation.SOFTMAX_BACKWARD,
                Operation.LOG_SOFTMAX_BACKWARD).contains(operation)) {
            require(first.equals(inputs.get(1).layout().shape()), "Backward seed shape");
        }
        if (operation == Operation.SOFTMAX || operation == Operation.LOG_SOFTMAX
                || operation == Operation.SOFTMAX_BACKWARD || operation == Operation.LOG_SOFTMAX_BACKWARD) {
            require(first.rank() > 0 && first.dimension(first.rank() - 1) > 0, "Softmax needs a nonempty last axis");
        }
        if (operation == Operation.DROPOUT) {
            double probability = attributes.number("p");
            require(Double.isFinite(probability) && probability >= 0 && probability < 1, "Dropout probability");
            require(attributes.integer("counter") >= 0, "Dropout counter");
            Math.addExact(attributes.integer("counter"), first.elements());
        }
        if (operation == Operation.LAYER_NORM || operation == Operation.RMS_NORM || operation == Operation.NORM_BACKWARD) {
            require(first.rank() > 0 && first.dimension(first.rank() - 1) > 0, "Normalization width");
            Shape width = Shape.of(first.dimension(first.rank() - 1));
            require(inputs.get(1).layout().shape().equals(width), "Normalization scale shape");
            boolean rms = operation == Operation.RMS_NORM || operation == Operation.NORM_BACKWARD && attributes.flag("rms");
            if (!rms) {
                require(inputs.get(2).layout().shape().equals(width), "Normalization bias shape");
            }
            if (operation == Operation.NORM_BACKWARD) {
                require(inputs.getLast().layout().shape().equals(first), "Normalization seed shape");
            }
            require(attributes.number("epsilon") > 0 && Double.isFinite(attributes.number("epsilon")), "Normalization epsilon");
        }
        if (operation == Operation.ROPE || operation == Operation.ROPE_BACKWARD) {
            require(first.rank() >= 2 && first.dimension(first.rank() - 1) > 0 && first.dimension(first.rank() - 1) % 2 == 0, "Rotary dimension");
            require(Double.isFinite(attributes.number("base")) && attributes.number("base") > 0, "Rotary base");
            if (inputs.size() == 2) {
                require(Shape.broadcast(inputs.get(1).layout().shape(), first.prefix(first.rank() - 1)).equals(first.prefix(first.rank() - 1)), "Position broadcasting");
            } else {
                require(attributes.integer("offset") >= 0, "Position offset");
                Math.addExact(attributes.integer("offset"), first.dimension(first.rank() - 2));
            }
        }
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new IllegalArgumentException(message);
        }
    }

    public static Shape reducedShape(Shape input, int[] axes, boolean keep) {
        boolean[] reduced = new boolean[input.rank()];
        for (int axis : axes) {
            int normalized = input.axis(axis);
            require(!reduced[normalized], "Duplicate reduction axis");
            reduced[normalized] = true;
        }
        long[] result = new long[keep ? input.rank() : input.rank() - axes.length];
        int at = 0;
        for (int i = 0; i < input.rank(); i++) {
            if (keep || !reduced[i]) {
                result[at++] = reduced[i] ? 1 : input.dimension(i);
            }
        }
        return Shape.of(result);
    }

    public static Shape gatherShape(Shape input, Shape indices, int requestedAxis) {
        int axis = input.axis(requestedAxis);
        return input.prefix(axis).append(indices.dimensions())
                .append(Arrays.copyOfRange(input.dimensions(), axis + 1, input.rank()));
    }

    public static Shape concatShape(List<Shape> shapes, int requestedAxis) {
        Shape first = shapes.getFirst();
        int axis = first.axis(requestedAxis);
        long size = 0;
        for (Shape shape : shapes) {
            require(shape.rank() == first.rank(), "Concat ranks differ");
            for (int i = 0; i < shape.rank(); i++) {
                require(i == axis || first.dimension(i) == shape.dimension(i), "Concat dimensions differ");
            }
            size = Math.addExact(size, shape.dimension(axis));
        }
        return first.replace(axis, size);
    }
}

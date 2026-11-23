package org.dynamisengine.nn.backend;

import java.util.List;
import java.util.Objects;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;

/** Immutable operation request. All operations overwrite, except explicit AXPY. */
public record KernelCall(Operation operation, List<KernelTensor> inputs, KernelTensor output, float scalar) {
    public KernelCall {
        Objects.requireNonNull(operation, "operation");
        inputs = List.copyOf(inputs);
        Objects.requireNonNull(output, "output");
        int count = switch (operation) {
            case FILL -> 0;
            case ADD, SUBTRACT, MULTIPLY, MATMUL -> 2;
            default -> 1;
        };
        if (inputs.size() != count) {
            throw new IllegalArgumentException(operation + " requires " + count + " inputs");
        }
        for (KernelTensor input : inputs) {
            if (!input.buffer().device().equals(output.buffer().device())
                    || input.buffer().dtype() != output.buffer().dtype()) {
                throw new IllegalArgumentException("Kernel operands must share device and dtype");
            }
            if (input.buffer() == output.buffer()) {
                throw new IllegalArgumentException("Kernel output must not alias inputs");
            }
        }
        if (operation != Operation.COPY && output.buffer().dtype() != DType.FP32) {
            throw new IllegalArgumentException(operation + " requires FP32");
        }
        Shape actual = output.layout().shape();
        Shape expected = switch (operation) {
            case FILL -> actual;
            case SUM -> Shape.of();
            case ADD, SUBTRACT, MULTIPLY -> Shape.broadcast(inputs.get(0).layout().shape(),
                    inputs.get(1).layout().shape());
            case MATMUL -> {
                Shape a = inputs.get(0).layout().shape();
                Shape b = inputs.get(1).layout().shape();
                if (a.rank() != 2 || b.rank() != 2 || a.dimension(1) != b.dimension(0)) {
                    throw new IllegalArgumentException("MATMUL requires compatible rank-two operands");
                }
                yield Shape.of(a.dimension(0), b.dimension(1));
            }
            case SUM_TO -> {
                if (!Shape.broadcast(actual, inputs.get(0).layout().shape())
                        .equals(inputs.get(0).layout().shape())) {
                    throw new IllegalArgumentException("Invalid broadcast reduction target");
                }
                yield actual;
            }
            default -> inputs.get(0).layout().shape();
        };
        if (!actual.equals(expected) || !output.layout().isContiguous()) {
            throw new IllegalArgumentException("Invalid output layout for " + operation);
        }
    }
}

package org.dynamisengine.nn.ops;

import org.dynamisengine.nn.backend.KernelAttributes;
import org.dynamisengine.nn.backend.KernelCall;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;

import java.util.Arrays;
import java.util.List;

final class MoreOps {
    private MoreOps() {
    }

    static Tensor divide(Tensor a, Tensor b) {
        return Ops.execute(Operation.DIVIDE, List.of(a, b), Shape.broadcast(a.shape(), b.shape()), 0, (dy, saved) -> {
            Tensor[] result = new Tensor[2];
            try {
                if (saved.get(0).requiresGrad()) {
                    try (Tensor g = Ops.divide(dy, saved.get(1))) {
                        result[0] = Ops.sumTo(g, saved.get(0).shape());
                    }
                }
                if (saved.get(1).requiresGrad()) {
                    try (Tensor square = Ops.square(saved.get(1)); Tensor product = Ops.multiply(dy, saved.get(0));
                         Tensor ratio = Ops.divide(product, square); Tensor negative = Ops.scale(ratio, -1)) {
                        result[1] = Ops.sumTo(negative, saved.get(1).shape());
                    }
                }
                return result;
            } catch (RuntimeException error) {
                close(result);
                throw error;
            }
        });
    }

    static Tensor unary(Tensor x, Operation op) {
        return Ops.execute(op, List.of(x), x.shape(), 0, (dy, saved) -> {
            Tensor input = saved.getFirst();
            return new Tensor[]{switch (op) {
                case EXP -> {
                    try (Tensor y = Ops.exp(input)) {
                        yield Ops.multiply(dy, y);
                    }
                }
                case LOG -> Ops.divide(dy, input);
                case SQRT -> {
                    try (Tensor y = Ops.sqrt(input); Tensor denominator = Ops.scale(y, 2)) {
                        yield Ops.divide(dy, denominator);
                    }
                }
                default -> throw new IllegalArgumentException("Unknown unary derivative");
            }};
        });
    }

    static KernelAttributes reduction(Shape shape, int[] axes, boolean keep) {
        int[] normalized = Arrays.stream(axes).map(shape::axis).sorted().toArray();
        KernelCall.reducedShape(shape, normalized, keep);
        return KernelAttributes.of("axes", Arrays.stream(normalized).boxed().toList()).and("keep", keep);
    }

    static Tensor reduce(Tensor x, int[] axes, boolean keep, Operation op) {
        KernelAttributes attrs = reduction(x.shape(), axes, keep);
        Shape target = KernelCall.reducedShape(x.shape(), attrs.axes(), keep);
        return Ops.execute(op, List.of(x), target, op == Operation.ARGMAX ? DType.INT32 : DType.FP32, 0, attrs,
                op == Operation.ARGMAX ? null : (dy, saved) -> {
                    if (op == Operation.REDUCE_MAX) {
                        return new Tensor[]{Ops.execute(Operation.MAX_BACKWARD, List.of(saved.getFirst(), dy),
                                saved.getFirst().shape(), DType.FP32, 0, attrs, null)};
                    }
                    Shape kept = KernelCall.reducedShape(saved.getFirst().shape(), attrs.axes(), true);
                    try (Tensor packed = Ops.contiguous(dy); Tensor shaped = Ops.reshape(packed, kept)) {
                        return new Tensor[]{Ops.broadcastTo(shaped, saved.getFirst().shape())};
                    }
                });
    }

    static Tensor mean(Tensor x, int[] axes, boolean keep) {
        KernelAttributes attrs = reduction(x.shape(), axes, keep);
        long count = 1;
        for (int axis : attrs.axes()) {
            count = Math.multiplyExact(count, x.shape().dimension(axis));
        }
        if (count == 0) {
            throw new IllegalArgumentException("Empty mean domain");
        }
        try (Tensor sum = reduce(x, axes, keep, Operation.REDUCE_SUM)) {
            return Ops.scale(sum, 1f / count);
        }
    }

    static Tensor permute(Tensor x, int... axes) {
        int[] normalized = Arrays.stream(axes).map(x.shape()::axis).toArray();
        var layout = x.layout().permute(normalized);
        int[] inverse = new int[axes.length];
        for (int i = 0; i < axes.length; i++) {
            inverse[normalized[i]] = i;
        }
        return Ops.view(x, layout, false, (dy, saved) -> new Tensor[]{Ops.permute(dy, inverse)});
    }

    static Tensor slice(Tensor x, int axis, long start, long end, long step) {
        int normalized = x.shape().axis(axis);
        var layout = x.layout().slice(normalized, start, end, step);
        KernelAttributes attrs = KernelAttributes.of("axis", normalized).and("start", start).and("end", end)
                .and("step", step).and("shape", x.shape());
        return Ops.view(x, layout, false, (dy, saved) -> new Tensor[]{
                Ops.execute(Operation.SLICE_BACKWARD, List.of(dy), saved.getFirst().shape(), DType.FP32, 0, attrs, null)
        });
    }

    static Tensor gather(Tensor x, int axis, Tensor indices) {
        int normalized = x.shape().axis(axis);
        KernelAttributes attrs = KernelAttributes.of("axis", normalized);
        return Ops.execute(Operation.GATHER, List.of(x, indices), KernelCall.gatherShape(x.shape(), indices.shape(), normalized),
                x.dtype(), 0, attrs, (dy, saved) -> new Tensor[]{
                        scatter(dy, normalized, saved.get(1), saved.get(0).shape()), null
                });
    }

    static Tensor scatter(Tensor values, int axis, Tensor indices, Shape target) {
        int normalized = target.axis(axis);
        return Ops.execute(Operation.SCATTER_ADD, List.of(values, indices), target, DType.FP32, 0,
                KernelAttributes.of("axis", normalized), (dy, saved) -> new Tensor[]{gather(dy, normalized, saved.get(1)), null});
    }

    static Tensor concat(int axis, List<Tensor> inputs) {
        if (inputs.isEmpty()) {
            throw new IllegalArgumentException("Concat needs at least one tensor");
        }
        int normalized = inputs.getFirst().shape().axis(axis);
        Shape target = KernelCall.concatShape(inputs.stream().map(Tensor::shape).toList(), normalized);
        return Ops.execute(Operation.CONCAT, inputs, target, inputs.getFirst().dtype(), 0,
                KernelAttributes.of("axis", normalized), (dy, saved) -> {
                    Tensor[] result = new Tensor[saved.size()];
                    long start = 0;
                    try {
                        for (int i = 0; i < saved.size(); i++) {
                            long end = start + saved.get(i).shape().dimension(normalized);
                            if (saved.get(i).requiresGrad()) {
                                result[i] = Ops.slice(dy, normalized, start, end);
                            }
                            start = end;
                        }
                        return result;
                    } catch (RuntimeException error) {
                        close(result);
                        throw error;
                    }
                });
    }

    static Tensor cast(Tensor x, DType dtype) {
        return dtype == x.dtype() ? Ops.copy(x) : Ops.execute(Operation.CAST, List.of(x), x.shape(), dtype, 0, KernelAttributes.EMPTY, null);
    }

    static Tensor condition(Tensor a, Tensor b, Operation op) {
        return Ops.execute(op, List.of(a, b), Shape.broadcast(a.shape(), b.shape()), DType.BOOL, 0, KernelAttributes.EMPTY, null);
    }

    static Tensor where(Tensor condition, Tensor a, Tensor b) {
        Shape target = Shape.broadcast(condition.shape(), Shape.broadcast(a.shape(), b.shape()));
        return Ops.execute(Operation.WHERE, List.of(condition, a, b), target, a.dtype(), 0, KernelAttributes.EMPTY,
                (dy, saved) -> {
                    Tensor[] result = new Tensor[3];
                    try (Tensor zero = dy.context().full(Shape.of(), 0)) {
                        if (saved.get(1).requiresGrad()) {
                            try (Tensor selected = Ops.where(saved.get(0), dy, zero)) {
                                result[1] = Ops.sumTo(selected, saved.get(1).shape());
                            }
                        }
                        if (saved.get(2).requiresGrad()) {
                            try (Tensor selected = Ops.where(saved.get(0), zero, dy)) {
                                result[2] = Ops.sumTo(selected, saved.get(2).shape());
                            }
                        }
                        return result;
                    } catch (RuntimeException error) {
                        close(result);
                        throw error;
                    }
                });
    }

    private static void close(Tensor[] tensors) {
        for (Tensor tensor : tensors) {
            if (tensor != null) {
                tensor.close();
            }
        }
    }
}

package org.dynamisengine.nn.tensor;

import java.util.Objects;

/**
 * Nonnegative element strides and an element offset into shared storage.
 */
public final class TensorLayout {
    private final Shape shape;
    private final long[] strides;
    private final long offset;

    public TensorLayout(Shape shape, long[] strides, long offset) {
        this.shape = Objects.requireNonNull(shape, "shape");
        this.strides = Objects.requireNonNull(strides, "strides").clone();
        if (strides.length != shape.rank() || offset < 0) {
            throw new IllegalArgumentException("Invalid layout rank or offset");
        }
        for (long stride : strides) {
            if (stride < 0) {
                throw new IllegalArgumentException("Negative strides are not supported");
            }
        }
        this.offset = offset;
        requiredElements();
    }

    public static TensorLayout contiguous(Shape shape) {
        long[] strides = new long[shape.rank()];
        if (shape.elements() == 0) {
            return new TensorLayout(shape, strides, 0);
        }
        long stride = 1;
        try {
            for (int axis = shape.rank() - 1; axis >= 0; axis--) {
                strides[axis] = stride;
                stride = Math.multiplyExact(stride, shape.dimension(axis));
            }
        } catch (ArithmeticException error) {
            throw new IllegalArgumentException("Layout strides overflow long", error);
        }
        return new TensorLayout(shape, strides, 0);
    }

    public Shape shape() {
        return shape;
    }

    public long stride(int axis) {
        return strides[axis];
    }

    public long offset() {
        return offset;
    }

    public long[] strides() {
        return strides.clone();
    }

    public TensorLayout permute(int... axes) {
        if (axes.length != shape.rank()) {
            throw new IllegalArgumentException("Permutation rank mismatch");
        }
        boolean[] used = new boolean[axes.length];
        long[] dimensions = new long[axes.length];
        long[] result = new long[axes.length];
        for (int i = 0; i < axes.length; i++) {
            int axis = shape.axis(axes[i]);
            if (used[axis]) {
                throw new IllegalArgumentException("Duplicate permutation axis");
            }
            used[axis] = true;
            dimensions[i] = shape.dimension(axis);
            result[i] = strides[axis];
        }
        return new TensorLayout(Shape.of(dimensions), result, offset);
    }

    public TensorLayout slice(int requestedAxis, long start, long end, long step) {
        int axis = shape.axis(requestedAxis);
        if (step <= 0 || start < 0 || end < start || end > shape.dimension(axis)) {
            throw new IllegalArgumentException("Slice requires 0 <= start <= end <= dimension and positive step");
        }
        long length = end == start ? 0 : 1 + (end - start - 1) / step;
        Shape sliced = shape.replace(axis, length);
        long[] result = strides();
        try {
            result[axis] = Math.multiplyExact(result[axis], step);
            long first = sliced.elements() == 0 ? offset : Math.addExact(offset, Math.multiplyExact(start, strides[axis]));
            return new TensorLayout(sliced, result, first);
        } catch (ArithmeticException error) {
            throw new IllegalArgumentException("Slice arithmetic overflow", error);
        }
    }

    public boolean isContiguous() {
        if (shape.elements() == 0) {
            return true;
        }
        long expected = 1;
        for (int axis = shape.rank() - 1; axis >= 0; axis--) {
            if (shape.dimension(axis) > 1 && strides[axis] != expected) {
                return false;
            }
            expected = Math.multiplyExact(expected, shape.dimension(axis));
        }
        return true;
    }

    public long requiredElements() {
        if (shape.elements() == 0) {
            return offset;
        }
        try {
            long last = offset;
            for (int axis = 0; axis < shape.rank(); axis++) {
                last = Math.addExact(last, Math.multiplyExact(shape.dimension(axis) - 1, strides[axis]));
            }
            return Math.addExact(last, 1);
        } catch (ArithmeticException error) {
            throw new IllegalArgumentException("Layout storage range overflows long", error);
        }
    }

    /**
     * Translate a logical row-major index to its storage element index.
     */
    public long index(long linearIndex) {
        if (linearIndex < 0 || linearIndex >= shape.elements()) {
            throw new IndexOutOfBoundsException("Logical index: " + linearIndex);
        }
        long result = offset;
        for (int axis = shape.rank() - 1; axis >= 0; axis--) {
            result += (linearIndex % shape.dimension(axis)) * strides[axis];
            linearIndex /= shape.dimension(axis);
        }
        return result;
    }

    public TensorLayout transpose(int first, int second) {
        Objects.checkIndex(first, shape.rank());
        Objects.checkIndex(second, shape.rank());
        long[] dimensions = shape.dimensions();
        long[] swapped = strides.clone();
        long d = dimensions[first];
        dimensions[first] = dimensions[second];
        dimensions[second] = d;
        long s = swapped[first];
        swapped[first] = swapped[second];
        swapped[second] = s;
        return new TensorLayout(Shape.of(dimensions), swapped, offset);
    }

    public TensorLayout reshape(Shape target) {
        if (target.elements() != shape.elements() || !isContiguous()) {
            throw new IllegalArgumentException("Reshape needs equal element counts and contiguous input");
        }
        TensorLayout layout = contiguous(target);
        return new TensorLayout(target, layout.strides, offset);
    }

    public TensorLayout broadcastTo(Shape target) {
        if (!Shape.broadcast(shape, target).equals(target)) {
            throw new IllegalArgumentException("Cannot broadcast " + shape + " to " + target);
        }
        long[] expanded = new long[target.rank()];
        int shift = target.rank() - shape.rank();
        for (int axis = 0; axis < target.rank(); axis++) {
            int source = axis - shift;
            expanded[axis] = source < 0 || shape.dimension(source) == 1 ? 0 : strides[source];
        }
        return new TensorLayout(target, expanded, offset);
    }
}

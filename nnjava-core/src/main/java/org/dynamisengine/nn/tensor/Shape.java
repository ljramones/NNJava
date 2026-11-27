package org.dynamisengine.nn.tensor;

import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable dimensions with checked element counts. An empty dimension list is a scalar.
 */
public final class Shape {
    private final long[] dimensions;
    private final long elements;

    private Shape(long[] dimensions) {
        this.dimensions = dimensions.clone();
        boolean empty = false;
        for (long dimension : dimensions) {
            if (dimension < 0) {
                throw new IllegalArgumentException("Negative dimension: " + dimension);
            }
            empty |= dimension == 0;
        }
        long count = empty ? 0 : 1;
        try {
            if (!empty) {
                for (long dimension : dimensions) {
                    count = Math.multiplyExact(count, dimension);
                }
            }
        } catch (ArithmeticException error) {
            throw new IllegalArgumentException("Shape element count overflows long", error);
        }
        this.elements = count;
    }

    public static Shape of(long... dimensions) {
        return new Shape(Objects.requireNonNull(dimensions, "dimensions"));
    }

    public static Shape matmul(Shape a, Shape b) {
        if (a.rank() < 2 || b.rank() < 2 || a.dimension(a.rank() - 1) != b.dimension(b.rank() - 2)) {
            throw new IllegalArgumentException("matmul requires rank >= 2 and equal inner dimensions");
        }
        return broadcast(a.prefix(a.rank() - 2), b.prefix(b.rank() - 2))
                .append(a.dimension(a.rank() - 2), b.dimension(b.rank() - 1));
    }

    public static Shape broadcast(Shape left, Shape right) {
        int rank = Math.max(left.rank(), right.rank());
        long[] result = new long[rank];
        for (int axis = 0; axis < rank; axis++) {
            int la = axis - (rank - left.rank());
            int ra = axis - (rank - right.rank());
            long l = la < 0 ? 1 : left.dimension(la);
            long r = ra < 0 ? 1 : right.dimension(ra);
            if (l != r && l != 1 && r != 1) {
                throw new IllegalArgumentException("Cannot broadcast " + left + " and " + right);
            }
            result[axis] = l == 1 ? r : l;
        }
        return of(result);
    }

    public int rank() {
        return dimensions.length;
    }

    public long dimension(int axis) {
        return dimensions[axis];
    }

    public long elements() {
        return elements;
    }

    public long[] dimensions() {
        return dimensions.clone();
    }

    public int axis(int axis) {
        int normalized = axis < 0 ? axis + rank() : axis;
        if (normalized < 0 || normalized >= rank()) {
            throw new IllegalArgumentException("Axis " + axis + " outside rank " + rank());
        }
        return normalized;
    }

    public Shape prefix(int length) {
        if (length < 0 || length > rank()) {
            throw new IllegalArgumentException("Invalid prefix length");
        }
        return of(Arrays.copyOf(dimensions, length));
    }

    public Shape append(long... suffix) {
        long[] result = Arrays.copyOf(dimensions, rank() + suffix.length);
        System.arraycopy(suffix, 0, result, rank(), suffix.length);
        return of(result);
    }

    public Shape replace(int axis, long size) {
        long[] result = dimensions();
        result[axis(axis)] = size;
        return of(result);
    }

    @Override
    public boolean equals(Object other) {
        return other instanceof Shape shape && Arrays.equals(dimensions, shape.dimensions);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(dimensions);
    }

    @Override
    public String toString() {
        return Arrays.toString(dimensions);
    }
}

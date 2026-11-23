package org.dynamisengine.nn.tensor;

import java.util.Arrays;
import java.util.Objects;

/** Immutable dimensions with checked element counts. An empty dimension list is a scalar. */
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

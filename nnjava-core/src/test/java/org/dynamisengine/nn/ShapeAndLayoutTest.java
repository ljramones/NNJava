package org.dynamisengine.nn;

import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.TensorLayout;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ShapeAndLayoutTest {
    @Test
    void shapesAreImmutableAndBroadcastTrailingDimensionsIncludingZero() {
        long[] dimensions = {2, 3};
        Shape shape = Shape.of(dimensions);
        dimensions[0] = 99;
        shape.dimensions()[1] = 99;
        assertEquals(Shape.of(2, 3), shape);
        assertEquals(1, Shape.of().elements());
        assertEquals(Shape.of(2, 4, 3), Shape.broadcast(Shape.of(2, 1, 3), Shape.of(4, 3)));
        assertEquals(Shape.of(2, 0, 3), Shape.broadcast(Shape.of(2, 0, 3), Shape.of(1, 3)));
        assertThrows(IllegalArgumentException.class, () -> Shape.broadcast(Shape.of(2), Shape.of(3)));
    }

    @Test
    void rejectsNegativeDimensionsAndCheckedArithmeticOverflow() {
        assertThrows(IllegalArgumentException.class, () -> Shape.of(-1));
        assertThrows(IllegalArgumentException.class, () -> Shape.of(Long.MAX_VALUE, 2));
        assertEquals(0, Shape.of(Long.MAX_VALUE, 0, Long.MAX_VALUE).elements());
        assertEquals(0, TensorLayout.contiguous(Shape.of(0, Long.MAX_VALUE, Long.MAX_VALUE)).requiredElements());
        assertThrows(IllegalArgumentException.class,
                () -> new TensorLayout(Shape.of(2), new long[]{Long.MAX_VALUE}, 1));
        assertThrows(IllegalArgumentException.class,
                () -> new TensorLayout(Shape.of(2), new long[]{-1}, 0));
    }

    @Test
    void transposeAndBroadcastIndexSharedStorage() {
        TensorLayout transposed = TensorLayout.contiguous(Shape.of(2, 3)).transpose(0, 1);
        long[] expected = {0, 3, 1, 4, 2, 5};
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], transposed.index(i));
        }
        assertFalse(transposed.isContiguous());
        assertThrows(IllegalArgumentException.class, () -> transposed.reshape(Shape.of(6)));
        TensorLayout broadcast = TensorLayout.contiguous(Shape.of(3)).broadcastTo(Shape.of(2, 3));
        assertEquals(0, broadcast.stride(0));
        assertEquals(1, broadcast.index(4));
        assertThrows(IndexOutOfBoundsException.class, () -> broadcast.index(6));
    }

    @Test
    void layoutsSupportScalarOffsetsAndRejectInvalidBroadcastOrAxes() {
        TensorLayout scalar = new TensorLayout(Shape.of(), new long[0], 3);
        assertEquals(3, scalar.index(0));
        assertEquals(4, scalar.requiredElements());
        assertEquals(0, TensorLayout.contiguous(Shape.of(2, 0)).requiredElements());
        assertThrows(IllegalArgumentException.class,
                () -> TensorLayout.contiguous(Shape.of(2, 3)).broadcastTo(Shape.of(3)));
        assertThrows(IndexOutOfBoundsException.class,
                () -> TensorLayout.contiguous(Shape.of(2, 3)).transpose(0, 2));
    }
}

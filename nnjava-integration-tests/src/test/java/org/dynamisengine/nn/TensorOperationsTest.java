package org.dynamisengine.nn;

import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class TensorOperationsTest {
    @Test
    void hostArraysAreCopiedAndAllStorageTypesRoundTrip() {
        try (var context = ExecutionContext.cpu()) {
            float[] original = {1, 2};
            try (var floats = context.tensor(original, Shape.of(2));
                 var ints = context.tensor(new int[]{7, -4}, Shape.of(2));
                 var bools = context.tensor(new boolean[]{true, false}, Shape.of(2));
                 var copied = Ops.copy(ints)) {
                original[0] = 99;
                floats.toFloatArray()[1] = 99;
                assertArrayEquals(new float[]{1, 2}, floats.toFloatArray());
                assertArrayEquals(new int[]{7, -4}, copied.toIntArray());
                assertArrayEquals(new boolean[]{true, false}, bools.toBooleanArray());
                assertThrows(IllegalArgumentException.class, floats::toIntArray);
                assertThrows(IllegalArgumentException.class, () -> Ops.add(ints, ints));
            }
            assertEquals(0, context.liveAllocations());
        }
    }

    @Test
    void broadcastAndStridedOperationsGiveExpectedResults() {
        try (var context = ExecutionContext.cpu();
             var a = context.tensor(new float[]{1, 2, 3, 4, 5, 6}, Shape.of(2, 3));
             var b = context.tensor(new float[]{10, 20, 30}, Shape.of(3));
             var added = Ops.add(a, b);
             var transposed = Ops.transpose(a);
             var doubled = Ops.add(transposed, transposed);
             var summed = Ops.sum(transposed);
             var reduced = Ops.sumTo(a, Shape.of(3))) {
            assertArrayEquals(new float[]{11, 22, 33, 14, 25, 36}, added.toFloatArray());
            assertArrayEquals(new float[]{2, 8, 4, 10, 6, 12}, doubled.toFloatArray());
            assertEquals(21f, summed.item());
            assertArrayEquals(new float[]{5, 7, 9}, reduced.toFloatArray());
        }
    }

    @Test
    void matrixMultiplicationHandlesRectanglesAndTransposedInputs() {
        try (var context = ExecutionContext.cpu();
             var a = context.tensor(new float[]{1, 2, 3, 4, 5, 6}, Shape.of(2, 3));
             var b = context.tensor(new float[]{7, 8, 9, 10, 11, 12}, Shape.of(3, 2));
             var result = Ops.matmul(a, b);
             var at = Ops.transpose(a);
             var bt = Ops.transpose(b);
             var reversed = Ops.matmul(bt, at);
             var gram = Ops.matmul(at, a)) {
            assertArrayEquals(new float[]{58, 64, 139, 154}, result.toFloatArray());
            assertArrayEquals(new float[]{58, 139, 64, 154}, reversed.toFloatArray());
            assertArrayEquals(new float[]{17, 22, 27, 22, 29, 36, 27, 36, 45}, gram.toFloatArray());
        }
    }

    @Test
    void broadcastedMatricesCanBePackedForGemm() {
        try (var context = ExecutionContext.cpu();
             var row = context.tensor(new float[]{1, 2, 3}, Shape.of(1, 3));
             var expanded = Ops.broadcastTo(row, Shape.of(2, 3));
             var column = context.tensor(new float[]{2, 3, 4}, Shape.of(3, 1));
             var result = Ops.matmul(expanded, column)) {
            assertArrayEquals(new float[]{20, 20}, result.toFloatArray());
        }
    }

    @Test
    void viewsOutliveOriginalHandleAndBroadcastViewsAreReadOnly() {
        try (var context = ExecutionContext.cpu()) {
            Tensor source = context.tensor(new float[]{1, 2, 3, 4}, Shape.of(2, 2));
            try (var view = Ops.transpose(source); var detached = source.detach()) {
                source.close();
                source.close();
                assertArrayEquals(new float[]{1, 3, 2, 4}, view.toFloatArray());
                assertThrows(IllegalStateException.class, source::toFloatArray);
                context.fill(detached, 7);
                assertArrayEquals(new float[]{7, 7, 7, 7}, view.toFloatArray());
            }
            assertEquals(0, context.liveAllocations());
            try (var scalar = context.full(Shape.of(), 1); var expanded = Ops.broadcastTo(scalar, Shape.of(3))) {
                assertThrows(IllegalArgumentException.class, () -> context.fill(expanded, 2));
                assertEquals(1f, scalar.item());
            }
        }
    }

    @Test
    void emptyReductionsAndZeroInnerDimensionHaveDefinedResults() {
        try (var context = ExecutionContext.cpu();
             var empty = context.zeros(DType.FP32, Shape.of(0));
             var total = Ops.sum(empty);
             var a = context.empty(DType.FP32, Shape.of(2, 0));
             var b = context.empty(DType.FP32, Shape.of(0, 3));
             var product = Ops.matmul(a, b)) {
            assertEquals(0f, total.item());
            assertThrows(IllegalArgumentException.class, () -> Ops.mean(empty));
            assertArrayEquals(new float[6], product.toFloatArray());
        }
    }

    @Test
    void structuralErrorsDoNotPoisonTheContextOrLeakOutputs() {
        try (var first = ExecutionContext.cpu(); var second = ExecutionContext.cpu();
             var a = first.full(Shape.of(2), 1); var b = first.full(Shape.of(3), 2);
             var foreign = second.full(Shape.of(2), 1)) {
            assertThrows(IllegalArgumentException.class, () -> Ops.add(a, b));
            assertThrows(IllegalArgumentException.class, () -> Ops.add(a, foreign));
            assertThrows(IllegalArgumentException.class, () -> Ops.matmul(a, a));
            assertEquals(2, first.liveAllocations());
            try (var valid = Ops.sum(a)) {
                assertEquals(2f, valid.item());
            }
        }
    }

    @Test
    void closingContextInvalidatesTensorsAndAllowsLaterHandleRelease() {
        var context = ExecutionContext.cpu();
        Tensor value = context.full(Shape.of(), 3);
        context.close();
        context.close();
        assertThrows(IllegalStateException.class, value::item);
        value.close();
        assertEquals(0, context.liveAllocations());
    }
}

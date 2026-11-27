package org.dynamisengine.nn.state;

import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;

/**
 * Immutable portable host copy, independent of a context's allocation lifetime.
 */
public final class TensorSnapshot {
    private final DType dtype;
    private final Shape shape;
    private final Object values;

    public TensorSnapshot(Shape shape, float[] values) {
        this(DType.FP32, shape, values.clone(), values.length);
    }

    public TensorSnapshot(Shape shape, int[] values) {
        this(DType.INT32, shape, values.clone(), values.length);
    }

    public TensorSnapshot(Shape shape, boolean[] values) {
        this(DType.BOOL, shape, values.clone(), values.length);
    }

    public TensorSnapshot(Shape shape, DType dtype, short[] values) {
        this(dtype, shape, values.clone(), values.length);
        if (!dtype.isLowPrecision()) {
            throw new IllegalArgumentException("Raw shorts require FP16 or BF16");
        }
    }

    private TensorSnapshot(DType dtype, Shape shape, Object values, int count) {
        if (shape.elements() != count) {
            throw new IllegalArgumentException("Snapshot shape/count mismatch");
        }
        this.dtype = dtype;
        this.shape = shape;
        this.values = values;
    }

    public static TensorSnapshot capture(Tensor tensor) {
        return switch (tensor.dtype()) {
            case FP32 -> owned(tensor.dtype(), tensor.shape(), tensor.toFloatArray());
            case INT32 -> owned(tensor.dtype(), tensor.shape(), tensor.toIntArray());
            case BOOL -> owned(tensor.dtype(), tensor.shape(), tensor.toBooleanArray());
            case FP16, BF16 -> owned(tensor.dtype(), tensor.shape(), tensor.toShortArray());
        };
    }

    static TensorSnapshot owned(DType dtype, Shape shape, Object values) {
        return new TensorSnapshot(dtype, shape, values, java.lang.reflect.Array.getLength(values));
    }

    public Object valuesCopy() {
        return switch (dtype) {
            case FP32 -> floats();
            case INT32 -> integers();
            case BOOL -> booleans();
            case FP16, BF16 -> shorts();
        };
    }

    public DType dtype() {
        return dtype;
    }

    public Shape shape() {
        return shape;
    }

    public float[] floats() {
        return ((float[]) values).clone();
    }

    public int[] integers() {
        return ((int[]) values).clone();
    }

    public boolean[] booleans() {
        return ((boolean[]) values).clone();
    }

    public short[] shorts() {
        return ((short[]) values).clone();
    }

    public Tensor materialize(ExecutionContext context) {
        return switch (dtype) {
            case FP32 -> context.tensor((float[]) values, shape);
            case INT32 -> context.tensor((int[]) values, shape);
            case BOOL -> context.tensor((boolean[]) values, shape);
            case FP16, BF16 -> context.tensor((short[]) values, dtype, shape);
        };
    }

    public void validate(Tensor target) {
        if (target.dtype() != dtype || !target.shape().equals(shape)) {
            throw new IllegalArgumentException("Snapshot does not match target tensor");
        }
    }

    public void assignTo(Tensor target) {
        validate(target);
        try (Tensor source = materialize(target.context())) {
            target.context().assign(target, source);
        }
    }
}

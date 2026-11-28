package org.dynamisengine.nn.formats;

import java.util.List;
import java.util.Objects;

/** Immutable row-major logical shape and absolute byte range in its owning file. */
public record TensorDescriptor(String name, TensorType type, List<Long> shape, long offset, long byteSize) {
    public TensorDescriptor {
        Objects.requireNonNull(name); Objects.requireNonNull(type); shape = List.copyOf(shape);
        if (name.isEmpty() || offset < 0 || byteSize < 0 || shape.size() > 32) { throw new IllegalArgumentException("Invalid tensor descriptor"); }
        long elements = 1;
        for (long dimension : shape) {
            if (dimension < 0) { throw new IllegalArgumentException("Negative dimension"); }
            elements = Math.multiplyExact(elements, dimension);
        }
        if (type.quantized() && !shape.isEmpty() && shape.getLast() % type.blockElements() != 0) {
            throw new IllegalArgumentException("Quantized row must contain complete blocks");
        }
        if (type.byteSize(elements) != byteSize) { throw new IllegalArgumentException("Tensor byte count mismatch: " + name); }
        Math.addExact(offset, byteSize);
    }
    public long elements() { long result = 1; for (long dim : shape) { result = Math.multiplyExact(result, dim); } return result; }
    public long[] dimensions() { return shape.stream().mapToLong(Long::longValue).toArray(); }
}

package org.dynamisengine.nn.data;

import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;

import java.util.LinkedHashMap;
import java.util.Map;

public final class TensorDataset implements Dataset {
    private final Map<String, Tensor> fields = new LinkedHashMap<>();
    private final long size;
    private boolean closed;

    public TensorDataset(Map<String, Tensor> tensors) {
        if (tensors.isEmpty()) {
            throw new IllegalArgumentException("Dataset needs fields");
        }
        long rows = -1;
        try {
            for (var entry : tensors.entrySet()) {
                Shape shape = entry.getValue().shape();
                if (shape.rank() == 0 || rows >= 0 && shape.dimension(0) != rows) {
                    throw new IllegalArgumentException("Dataset row counts differ");
                }
                rows = shape.dimension(0);
                fields.put(entry.getKey(), entry.getValue().detach());
            }
        } catch (RuntimeException error) {
            fields.values().forEach(Tensor::close);
            throw error;
        }
        size = rows;
    }

    @Override
    public long size() {
        ensureOpen();
        return size;
    }

    @Override
    public Batch batch(ExecutionContext context, int[] indices) {
        ensureOpen();
        for (int index : indices) {
            if (index < 0 || index >= size) {
                throw new IllegalArgumentException("Dataset index out of bounds");
            }
        }
        fields.values().forEach(context::requireContext);
        Map<String, Tensor> result = new LinkedHashMap<>();
        try (Tensor selected = context.tensor(indices, Shape.of(indices.length))) {
            for (var field : fields.entrySet()) {
                result.put(field.getKey(), Ops.gather(field.getValue(), 0, selected));
            }
            return new Batch(result);
        } catch (RuntimeException error) {
            result.values().forEach(Tensor::close);
            throw error;
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Dataset is closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            fields.values().forEach(Tensor::close);
            closed = true;
        }
    }
}

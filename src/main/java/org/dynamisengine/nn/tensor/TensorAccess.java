package org.dynamisengine.nn.tensor;

import org.dynamisengine.nn.backend.Buffer;
import org.dynamisengine.nn.backend.KernelTensor;
import org.dynamisengine.nn.runtime.ExecutionContext;

/** Internal bridge for the runtime and differentiation engine; not an application API. */
public final class TensorAccess {
    private TensorAccess() {
    }

    public static Tensor create(ExecutionContext context, Buffer buffer, Shape shape, boolean requiresGrad) {
        if (requiresGrad && buffer.dtype() != DType.FP32) {
            throw new IllegalArgumentException("Only FP32 tensors support gradients");
        }
        return new Tensor(new Tensor.State(context, new Tensor.Storage(buffer),
                TensorLayout.contiguous(shape), requiresGrad, false));
    }

    public static Tensor view(Tensor source, TensorLayout layout, boolean readOnly) {
        source.checkOpen();
        if (layout.requiredElements() > source.state.storage.buffer.elements()) {
            throw new IllegalArgumentException("View exceeds storage");
        }
        return new Tensor(new Tensor.State(source.state.context, source.state.storage, layout, false,
                readOnly || source.state.readOnly));
    }

    public static KernelTensor kernel(Tensor tensor) {
        tensor.checkOpen();
        return new KernelTensor(tensor.state.storage.buffer, tensor.state.layout);
    }

    public static void beforeWrite(Tensor tensor) {
        tensor.checkOpen();
        if (tensor.state.readOnly || !tensor.state.layout.isContiguous()) {
            throw new IllegalArgumentException("Mutation requires a writable contiguous tensor");
        }
        if (tensor.state.storage.savedValues != 0) {
            throw new IllegalStateException("Cannot mutate storage saved by an active gradient tape");
        }
        tensor.state.storage.version++;
    }

    public static void track(Tensor output, long tapeId) {
        output.checkOpen();
        output.state.requiresGrad = true;
        output.state.tapeId = tapeId;
    }

    public static long tapeId(Tensor tensor) {
        tensor.checkOpen();
        return tensor.state.tapeId;
    }

    public static void setGradient(Tensor tensor, Tensor gradient) {
        tensor.checkOpen();
        Tensor retained = gradient.retain();
        if (tensor.state.gradient != null) {
            tensor.state.gradient.close();
        }
        tensor.state.gradient = retained;
    }

    public static Saved save(Tensor tensor) {
        return new Saved(tensor);
    }

    public static final class Saved implements AutoCloseable {
        private final Tensor tensor;
        private final long version;
        private boolean closed;

        private Saved(Tensor tensor) {
            this.tensor = tensor.retain();
            version = tensor.state.storage.version;
            tensor.state.storage.savedValues++;
        }

        public Tensor tensor() {
            if (closed || tensor.state.storage.version != version) {
                throw new IllegalStateException("Saved value was released or mutated");
            }
            return tensor;
        }

        @Override
        public void close() {
            if (!closed) {
                closed = true;
                tensor.state.storage.savedValues--;
                tensor.close();
            }
        }
    }
}

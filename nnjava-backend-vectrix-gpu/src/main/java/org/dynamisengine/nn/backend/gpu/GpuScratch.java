package org.dynamisengine.nn.backend.gpu;

import java.util.ArrayList;
import java.util.List;
import org.dynamisengine.gpu.api.compute.GpuCommandEncoder;
import org.dynamisengine.nn.backend.Buffer;
import org.dynamisengine.nn.backend.KernelTensor;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.TensorLayout;

/** Per-recording handles. Commands retain their native storage until execution completes or recording aborts. */
final class GpuScratch implements AutoCloseable {
    private final GpuBackend backend;
    private final List<Buffer> buffers = new ArrayList<>();
    GpuScratch(GpuBackend backend) { this.backend = backend; }
    KernelTensor tensor(Shape shape) { return tensor(DType.FP32, shape); }
    KernelTensor tensor(DType dtype, Shape shape) {
        var buffer = backend.allocate(dtype, shape.elements()); buffers.add(buffer);
        return new KernelTensor(buffer, TensorLayout.contiguous(shape));
    }
    KernelTensor packed(GpuCommandEncoder encoder, KernelTensor input, boolean zeroOffset) {
        if (input.layout().isContiguous() && (!zeroOffset || input.layout().offset() == 0)) { return input; }
        var output = tensor(input.buffer().dtype(), input.layout().shape());
        backend.copy(encoder, input, output); return output;
    }
    KernelTensor broadcast(GpuCommandEncoder encoder, KernelTensor input, Shape shape) {
        return packed(encoder, new KernelTensor(input.buffer(), input.layout().broadcastTo(shape)), false);
    }
    static KernelTensor flat(KernelTensor tensor, Shape shape) {
        if (!tensor.layout().isContiguous() || tensor.layout().shape().elements() != shape.elements()) {
            throw new IllegalArgumentException("Scratch reshape requires matching contiguous storage");
        }
        return new KernelTensor(tensor.buffer(), new TensorLayout(shape, TensorLayout.contiguous(shape).strides(), tensor.layout().offset()));
    }
    @Override public void close() { for (int i = buffers.size() - 1; i >= 0; i--) { buffers.get(i).close(); } }
}

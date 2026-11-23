package org.dynamisengine.nn.backend.vectrix;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import org.dynamisengine.nn.backend.Backend;
import org.dynamisengine.nn.backend.Buffer;
import org.dynamisengine.nn.backend.Completion;
import org.dynamisengine.nn.backend.KernelCall;
import org.dynamisengine.nn.backend.KernelTensor;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.runtime.Device;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.TensorLayout;
import org.dynamisengine.vectrix.blas.Blasf;
import org.dynamisengine.vectrix.blas.Gemmf;
import org.dynamisengine.vectrix.hash.Philox;

/** Synchronous array-backed CPU implementation. Scratch belongs to this context's backend. */
public final class VectrixBackend implements Backend {
    private final Device device = new Device("vectrix", "cpu");
    private final Set<CpuBuffer> allocations = new HashSet<>();
    private float[] scratch = new float[0];
    private float[] packedLeft = new float[0];
    private float[] packedRight = new float[0];
    private boolean closed;

    @Override
    public Device device() {
        return device;
    }

    @Override
    public boolean supports(Operation operation, DType dtype) {
        return operation == Operation.COPY || dtype == DType.FP32;
    }

    @Override
    public Buffer allocate(DType dtype, long elements) {
        ensureOpen();
        if (elements < 0 || elements > Integer.MAX_VALUE - 8L) {
            throw new IllegalArgumentException("CPU array element limit exceeded: " + elements);
        }
        CpuBuffer buffer = new CpuBuffer(dtype, (int) elements);
        allocations.add(buffer);
        return buffer;
    }

    @Override
    public Completion execute(KernelCall call) {
        ensureOpen();
        CpuBuffer out = cpu(call.output().buffer());
        for (KernelTensor input : call.inputs()) {
            cpu(input.buffer());
        }
        if (!supports(call.operation(), out.dtype)) {
            throw new UnsupportedOperationException("Vectrix CPU does not support " + call.operation() + "/" + out.dtype);
        }
        TensorLayout target = call.output().layout();
        int n = Math.toIntExact(target.shape().elements());
        int offset = Math.toIntExact(target.offset());
        if (call.operation() == Operation.COPY) {
            KernelTensor source = call.inputs().getFirst();
            copy(source, out.data, offset);
        } else if (call.operation() == Operation.FILL) {
            Arrays.fill(out.floats(), offset, offset + n, call.scalar());
        } else if (call.operation() == Operation.MATMUL) {
            matmul(call);
        } else if (call.operation() == Operation.SUM) {
            KernelTensor source = call.inputs().getFirst();
            float[] values = cpu(source.buffer()).floats();
            int count = Math.toIntExact(source.layout().shape().elements());
            if (source.layout().isContiguous()) {
                out.floats()[offset] = Blasf.sum(values, Math.toIntExact(source.layout().offset()), count);
            } else {
                scratch = capacity(scratch, count);
                for (int i = 0; i < count; i++) {
                    scratch[i] = values[Math.toIntExact(source.layout().index(i))];
                }
                out.floats()[offset] = Blasf.sum(scratch, 0, count);
            }
        } else if (call.operation() == Operation.SUM_TO) {
            KernelTensor source = call.inputs().getFirst();
            float[] values = cpu(source.buffer()).floats();
            float[] result = out.floats();
            Arrays.fill(result, offset, offset + n, 0f);
            TensorLayout destinations = target.broadcastTo(source.layout().shape());
            for (long i = 0; i < source.layout().shape().elements(); i++) {
                result[Math.toIntExact(destinations.index(i))] += values[Math.toIntExact(source.layout().index(i))];
            }
        } else if (call.operation() == Operation.AXPY) {
            KernelTensor source = call.inputs().getFirst();
            float[] values = cpu(source.buffer()).floats();
            if (source.layout().isContiguous()) {
                Blasf.axpy(call.scalar(), values, Math.toIntExact(source.layout().offset()), out.floats(), offset, n);
            } else {
                for (int i = 0; i < n; i++) {
                    out.floats()[offset + i] = Math.fma(call.scalar(),
                            values[Math.toIntExact(source.layout().index(i))], out.floats()[offset + i]);
                }
            }
        } else {
            KernelTensor first = call.inputs().getFirst();
            TensorLayout aLayout = first.layout().broadcastTo(target.shape());
            float[] a = cpu(first.buffer()).floats();
            KernelTensor second = call.inputs().size() == 2 ? call.inputs().get(1) : null;
            TensorLayout bLayout = second == null ? null : second.layout().broadcastTo(target.shape());
            float[] b = second == null ? null : cpu(second.buffer()).floats();
            for (int i = 0; i < n; i++) {
                float x = a[Math.toIntExact(aLayout.index(i))];
                float y = second == null ? 0f : b[Math.toIntExact(bLayout.index(i))];
                out.floats()[offset + i] = switch (call.operation()) {
                    case ADD -> x + y;
                    case SUBTRACT -> x - y;
                    case MULTIPLY -> x * y;
                    case SCALE -> x * call.scalar();
                    default -> throw new UnsupportedOperationException("Unknown operation " + call.operation());
                };
            }
        }
        return Completion.completed();
    }

    private void matmul(KernelCall call) {
        Shape a = call.inputs().get(0).layout().shape();
        Shape b = call.inputs().get(1).layout().shape();
        if (call.output().layout().shape().elements() == 0) {
            return;
        }
        int m = Math.toIntExact(a.dimension(0));
        int k = Math.toIntExact(a.dimension(1));
        int n = Math.toIntExact(b.dimension(1));
        float[] result = cpu(call.output().buffer()).floats();
        int outOffset = Math.toIntExact(call.output().layout().offset());
        if (m == 0 || n == 0 || k == 0) {
            Arrays.fill(result, outOffset, outOffset + m * n, 0f);
            return;
        }
        Matrix left = matrix(call.inputs().get(0), true);
        Matrix right = matrix(call.inputs().get(1), false);
        scratch = capacity(scratch, Gemmf.scratchSize(left.transpose, right.transpose, m, n, k));
        Gemmf.gemm(left.transpose, right.transpose, m, n, k, 1f,
                left.data, left.offset, left.stride, right.data, right.offset, right.stride,
                0f, result, outOffset, n, scratch, 0);
    }

    private Matrix matrix(KernelTensor tensor, boolean left) {
        TensorLayout layout = tensor.layout();
        float[] data = cpu(tensor.buffer()).floats();
        if (layout.stride(1) == 1 && layout.stride(0) >= layout.shape().dimension(1)) {
            return new Matrix(data, Math.toIntExact(layout.offset()), Math.toIntExact(layout.stride(0)), false);
        }
        if (layout.stride(0) == 1 && layout.stride(1) >= layout.shape().dimension(0)) {
            return new Matrix(data, Math.toIntExact(layout.offset()), Math.toIntExact(layout.stride(1)), true);
        }
        int count = Math.toIntExact(layout.shape().elements());
        float[] packed = capacity(left ? packedLeft : packedRight, count);
        for (int i = 0; i < count; i++) {
            packed[i] = data[Math.toIntExact(layout.index(i))];
        }
        if (left) {
            packedLeft = packed;
        } else {
            packedRight = packed;
        }
        return new Matrix(packed, 0, Math.toIntExact(layout.shape().dimension(1)), false);
    }

    private record Matrix(float[] data, int offset, int stride, boolean transpose) {
    }

    private static float[] capacity(float[] current, int required) {
        return current.length >= required ? current : new float[required];
    }

    private void copy(KernelTensor source, Object target, int offset) {
        CpuBuffer input = cpu(source.buffer());
        int n = Math.toIntExact(source.layout().shape().elements());
        for (int i = 0; i < n; i++) {
            int index = Math.toIntExact(source.layout().index(i));
            switch (input.dtype) {
                case FP32 -> ((float[]) target)[offset + i] = input.floats()[index];
                case INT32 -> ((int[]) target)[offset + i] = ((int[]) input.data)[index];
                case BOOL -> ((boolean[]) target)[offset + i] = ((boolean[]) input.data)[index];
            }
        }
    }

    private Completion uploadArray(Buffer target, Object source, DType dtype, int length) {
        CpuBuffer buffer = cpu(target);
        if (buffer.dtype != dtype || buffer.elements != length) {
            throw new IllegalArgumentException("Upload dtype or element count mismatch");
        }
        System.arraycopy(source, 0, buffer.data, 0, length);
        return Completion.completed();
    }

    private Completion downloadArray(KernelTensor source, Object target, DType dtype, int length) {
        if (cpu(source.buffer()).dtype != dtype || source.layout().shape().elements() != length) {
            throw new IllegalArgumentException("Download dtype or element count mismatch");
        }
        copy(source, target, 0);
        return Completion.completed();
    }

    @Override public Completion upload(Buffer target, float[] source) {
        return uploadArray(target, source, DType.FP32, source.length);
    }
    @Override public Completion upload(Buffer target, int[] source) {
        return uploadArray(target, source, DType.INT32, source.length);
    }
    @Override public Completion upload(Buffer target, boolean[] source) {
        return uploadArray(target, source, DType.BOOL, source.length);
    }
    @Override public Completion download(KernelTensor source, float[] target) {
        return downloadArray(source, target, DType.FP32, target.length);
    }
    @Override public Completion download(KernelTensor source, int[] target) {
        return downloadArray(source, target, DType.INT32, target.length);
    }
    @Override public Completion download(KernelTensor source, boolean[] target) {
        return downloadArray(source, target, DType.BOOL, target.length);
    }

    @Override
    public Completion randomNormal(Buffer target, long seed, long start, float standardDeviation) {
        CpuBuffer buffer = cpu(target);
        if (buffer.dtype != DType.FP32 || !Float.isFinite(standardDeviation) || standardDeviation < 0) {
            throw new IllegalArgumentException("Normal initialization requires FP32 and finite nonnegative deviation");
        }
        Philox.fillNormal(seed, start, 0f, standardDeviation, buffer.floats(), 0, buffer.elements);
        return Completion.completed();
    }

    private CpuBuffer cpu(Buffer buffer) {
        ensureOpen();
        if (!(buffer instanceof CpuBuffer cpu) || cpu.owner != this || cpu.isClosed()) {
            throw new IllegalArgumentException("Buffer does not belong to this live Vectrix backend");
        }
        return cpu;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("Backend is closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            for (CpuBuffer buffer : Set.copyOf(allocations)) {
                buffer.close();
            }
            scratch = packedLeft = packedRight = new float[0];
            closed = true;
        }
    }

    private final class CpuBuffer implements Buffer {
        private final VectrixBackend owner = VectrixBackend.this;
        private final DType dtype;
        private final int elements;
        private Object data;

        private CpuBuffer(DType dtype, int elements) {
            this.dtype = dtype;
            this.elements = elements;
            data = switch (dtype) {
                case FP32 -> new float[elements];
                case INT32 -> new int[elements];
                case BOOL -> new boolean[elements];
            };
        }

        private float[] floats() {
            return (float[]) data;
        }

        @Override public Device device() { return device; }
        @Override public DType dtype() { return dtype; }
        @Override public long elements() { return elements; }
        @Override public boolean isClosed() { return data == null; }

        @Override
        public void close() {
            data = null;
            allocations.remove(this);
        }
    }
}

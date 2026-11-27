package org.dynamisengine.nn.backend.vectrix;

import org.dynamisengine.nn.backend.*;
import org.dynamisengine.nn.runtime.Device;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.TensorLayout;
import org.dynamisengine.vectrix.blas.Blasf;
import org.dynamisengine.vectrix.blas.Gemmf;
import org.dynamisengine.vectrix.hash.Philox;
import org.dynamisengine.vectrix.nn.Optim;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * Synchronous array-backed CPU implementation. Scratch belongs to this context's backend.
 */
public final class VectrixBackend implements Backend {
    @Override public long maxElements(DType dtype) { return Integer.MAX_VALUE - 8L; }
    private final Device device = new Device("vectrix", "cpu");
    private final Set<CpuBuffer> allocations = new HashSet<>();
    private float[] scratch = new float[0];
    private float[] packedLeft = new float[0];
    private float[] packedRight = new float[0];
    private boolean closed;

    private static float[] capacity(float[] current, int required) {
        return current.length >= required ? current : new float[required];
    }

    @Override
    public Device device() {
        return device;
    }

    @Override
    public boolean supports(Operation operation, DType dtype) {
        if (dtype.isLowPrecision()) {
            return operation == Operation.COPY;
        }
        return switch (operation) {
            case COPY, CAST, GATHER, CONCAT, WHERE -> true;
            case EQUAL, LESS, GREATER, AND, OR, NOT, IS_FINITE, ALL -> dtype == DType.BOOL;
            case ARGMAX -> dtype == DType.INT32;
            default -> dtype == DType.FP32;
        };
    }

    @Override
    public boolean supports(KernelCall call) {
        return supports(call.operation(), call.output().buffer().dtype())
                && (call.operation() == Operation.COPY
                || call.inputs().stream().noneMatch(t -> t.buffer().dtype().isLowPrecision()));
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
        if (!supports(call)) {
            throw new UnsupportedOperationException("Vectrix CPU does not support " + call.operation() + "/" + out.dtype);
        }
        if (call.operation().ordinal() > Operation.AXPY.ordinal()) {
            Object[] inputs = call.inputs().stream().map(input -> cpu(input.buffer()).data).toArray();
            CpuTensorKernels.execute(call, inputs, out.data);
            return Completion.completed();
        }
        TensorLayout target = call.output().layout();
        int n = Math.toIntExact(target.shape().elements());
        int offset = Math.toIntExact(target.offset());
        if (call.operation() == Operation.COPY) {
            KernelTensor source = call.inputs().getFirst();
            for (long i = 0; i < target.shape().elements(); i++) {
                if (out.dtype.isLowPrecision()) {
                    ((short[]) out.data)[Math.toIntExact(target.index(i))] = ((short[]) cpu(source.buffer()).data)[Math.toIntExact(source.layout().index(i))];
                } else {
                    CpuTensorKernels.write(out.data, target.index(i), CpuTensorKernels.read(cpu(source.buffer()).data, source.layout(), i));
                }
            }
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

    @Override
    public Completion update(OptimizerCall call) {
        ensureOpen();
        var settings = call.settings();
        float[] parameter = cpu(call.parameter().buffer()).floats();
        float[] gradient = CpuTensorKernels.floats(call.gradient(), cpu(call.gradient().buffer()).data);
        int offset = Math.toIntExact(call.parameter().layout().offset());
        int n = Math.toIntExact(call.parameter().layout().shape().elements());
        float[] first = cpu(call.slots().get(0).buffer()).floats();
        int firstOffset = Math.toIntExact(call.slots().get(0).layout().offset());
        if (settings.kind() == OptimizerCall.Kind.ADAMW) {
            float[] second = cpu(call.slots().get(1).buffer()).floats();
            Optim.adamW(parameter, offset, gradient, 0, first, firstOffset, second,
                    Math.toIntExact(call.slots().get(1).layout().offset()), n, settings.learningRate(), settings.beta1(),
                    settings.beta2(), settings.epsilon(), settings.weightDecay(), settings.step());
        } else {
            Optim.sgd(parameter, offset, gradient, 0, first, firstOffset, n, settings.learningRate(), settings.beta1(),
                    settings.weightDecay(), settings.nesterov());
        }
        return Completion.completed();
    }

    private void matmul(KernelCall call) {
        Shape target = call.output().layout().shape();
        if (target.rank() == 2) {
            matrixProduct(call);
            return;
        }
        if (target.elements() == 0) {
            return;
        }
        long m = target.dimension(target.rank() - 2);
        long n = target.dimension(target.rank() - 1);
        Shape batch = target.prefix(target.rank() - 2);
        long k = call.inputs().getFirst().layout().shape().dimension(call.inputs().getFirst().layout().shape().rank() - 1);
        if (k == 0) {
            java.util.Arrays.fill(cpu(call.output().buffer()).floats(), 0f);
            return;
        }
        TensorLayout left = call.inputs().get(0).layout().broadcastTo(batch.append(m, k));
        TensorLayout right = call.inputs().get(1).layout().broadcastTo(batch.append(k, n));
        for (long b = 0; b < batch.elements(); b++) {
            long[] coordinates = CpuTensorKernels.coordinates(batch, b);
            long aOffset = left.offset();
            long bOffset = right.offset();
            for (int axis = 0; axis < coordinates.length; axis++) {
                aOffset += coordinates[axis] * left.stride(axis);
                bOffset += coordinates[axis] * right.stride(axis);
            }
            int r = target.rank();
            KernelTensor a = new KernelTensor(call.inputs().get(0).buffer(), new TensorLayout(Shape.of(m, k), new long[]{left.stride(r - 2), left.stride(r - 1)}, aOffset));
            KernelTensor v = new KernelTensor(call.inputs().get(1).buffer(), new TensorLayout(Shape.of(k, n), new long[]{right.stride(r - 2), right.stride(r - 1)}, bOffset));
            KernelTensor output = new KernelTensor(call.output().buffer(), new TensorLayout(Shape.of(m, n), new long[]{n, 1}, call.output().layout().offset() + b * m * n));
            matrixProduct(new KernelCall(Operation.MATMUL, java.util.List.of(a, v), output, 0));
        }
    }

    private void matrixProduct(KernelCall call) {
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

    private void copy(KernelTensor source, Object target, int offset) {
        CpuBuffer input = cpu(source.buffer());
        int n = Math.toIntExact(source.layout().shape().elements());
        for (int i = 0; i < n; i++) {
            int index = Math.toIntExact(source.layout().index(i));
            switch (input.dtype) {
                case FP32 -> ((float[]) target)[offset + i] = input.floats()[index];
                case INT32 -> ((int[]) target)[offset + i] = ((int[]) input.data)[index];
                case BOOL -> ((boolean[]) target)[offset + i] = ((boolean[]) input.data)[index];
                case FP16, BF16 -> ((short[]) target)[offset + i] = ((short[]) input.data)[index];
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

    @Override
    public Completion upload(Buffer target, float[] source) {
        return uploadArray(target, source, DType.FP32, source.length);
    }

    @Override
    public Completion upload(Buffer target, int[] source) {
        return uploadArray(target, source, DType.INT32, source.length);
    }

    @Override
    public Completion upload(Buffer target, boolean[] source) {
        return uploadArray(target, source, DType.BOOL, source.length);
    }

    @Override
    public Completion upload(Buffer target, short[] source) {
        if (!target.dtype().isLowPrecision()) {
            throw new IllegalArgumentException("Raw shorts require FP16/BF16");
        }
        return uploadArray(target, source, target.dtype(), source.length);
    }

    @Override
    public Completion download(KernelTensor source, float[] target) {
        return downloadArray(source, target, DType.FP32, target.length);
    }

    @Override
    public Completion download(KernelTensor source, int[] target) {
        return downloadArray(source, target, DType.INT32, target.length);
    }

    @Override
    public Completion download(KernelTensor source, boolean[] target) {
        return downloadArray(source, target, DType.BOOL, target.length);
    }

    @Override
    public Completion download(KernelTensor source, short[] target) {
        if (!source.buffer().dtype().isLowPrecision()) {
            throw new IllegalArgumentException("Raw shorts require FP16/BF16");
        }
        return downloadArray(source, target, source.buffer().dtype(), target.length);
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

    private record Matrix(float[] data, int offset, int stride, boolean transpose) {
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
                case FP16, BF16 -> new short[elements];
            };
        }

        private float[] floats() {
            return (float[]) data;
        }

        @Override
        public Device device() {
            return device;
        }

        @Override
        public DType dtype() {
            return dtype;
        }

        @Override
        public long elements() {
            return elements;
        }

        @Override
        public boolean isClosed() {
            return data == null;
        }

        @Override
        public void close() {
            data = null;
            allocations.remove(this);
        }
    }
}

package org.dynamisengine.nn.backend.gpu;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Consumer;

import org.dynamisengine.gpu.api.compute.BufferDescriptor;
import org.dynamisengine.gpu.api.compute.BufferUsage;
import org.dynamisengine.gpu.api.compute.GpuBuffer;
import org.dynamisengine.gpu.api.compute.GpuCommandEncoder;
import org.dynamisengine.gpu.api.compute.GpuDevice;
import org.dynamisengine.gpu.api.compute.Placement;
import org.dynamisengine.gpu.api.compute.kit.AccuracyContract;
import org.dynamisengine.gpu.api.compute.kit.StridedView;
import org.dynamisengine.gpu.primitives.Elementwise;
import org.dynamisengine.gpu.primitives.GemmF16;
import org.dynamisengine.gpu.primitives.Precision;
import org.dynamisengine.nn.backend.Backend;
import org.dynamisengine.nn.backend.Buffer;
import org.dynamisengine.nn.backend.Completion;
import org.dynamisengine.nn.backend.KernelCall;
import org.dynamisengine.nn.backend.KernelTensor;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.runtime.Device;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.TensorLayout;
import org.dynamisengine.vectrix.compute.GpuBlasf;
import org.dynamisengine.vectrix.compute.GpuGemmf;
import org.dynamisengine.vectrix.compute.GpuPointwise;

/**
 * Initial GPU storage and arithmetic adapter. Owns the supplied device. Coverage
 * is deliberately capability-gated: this is not yet a complete NN inference or
 * training backend. All numerical kernels come from the released upstream stack.
 */
public final class GpuBackend implements Backend {
    @Override public long maxElements(DType dtype) { return Integer.MAX_VALUE; }
    @Override public boolean supportsNormalInitialization() { return false; }
    private final GpuDevice gpu;
    private final GpuMode mode;
    private final Device device;
    private final Set<Allocation> allocations = new HashSet<>();
    private final List<GpuCompletion> pending = new ArrayList<>();
    private final Map<Operation, String> tiers = new LinkedHashMap<>();
    private GpuPointwise pointwise;
    private GpuBlasf blas;
    private GpuGemmf exactGemm;
    private Elementwise elementwise;
    private GemmF16 throughputGemm;
    private boolean closed;

    public GpuBackend(GpuDevice gpu, GpuMode mode) {
        this.gpu = Objects.requireNonNull(gpu, "gpu");
        this.mode = Objects.requireNonNull(mode, "mode");
        device = new Device(mode == GpuMode.EXACT ? "vectrix-gpu-exact" : "dynamis-gpu-throughput", GpuProviders.key(gpu.adapter()));
    }

    private static boolean whole(KernelTensor tensor) {
        return tensor.layout().isContiguous() && tensor.layout().offset() == 0
                && tensor.layout().shape().elements() == tensor.buffer().elements();
    }

    private static boolean matrixLayout(TensorLayout layout) {
        return layout.shape().rank() == 2 && (layout.stride(1) == 1
                && layout.stride(0) >= layout.shape().dimension(1)
                || layout.stride(0) == 1 && layout.stride(1) >= layout.shape().dimension(0));
    }

    private static StridedView view(TensorLayout layout) {
        return layout.shape().rank() == 0 ? StridedView.of(layout.offset(), new long[]{1}, new long[]{1})
                : StridedView.of(layout.offset(), layout.shape().dimensions(), layout.strides());
    }

    private static int width(DType dtype) {
        return dtype == DType.BOOL ? 4 : dtype.bytes();
    }

    @Override
    public Device device() {
        return device;
    }

    public GpuMode mode() {
        return mode;
    }

    /**
     * The last selected kernel tier for each executed operation.
     */
    public Map<Operation, String> executedTiers() {
        return Map.copyOf(tiers);
    }

    /**
     * Present after the first throughput GEMM; exposes the upstream accuracy contract.
     */
    public java.util.Optional<AccuracyContract> matmulAccuracy() {
        return throughputGemm == null ? java.util.Optional.empty() : java.util.Optional.of(throughputGemm.contract());
    }

    @Override
    public boolean supports(Operation operation, DType dtype) {
        if (operation == Operation.COPY) {
            return true;
        }
        if (dtype != DType.FP32) {
            return false;
        }
        return switch (operation) {
            case FILL, ADD, SUBTRACT, MULTIPLY, DIVIDE, EXP, LOG, SQRT, SCALE, AXPY, MATMUL -> true;
            default -> false;
        };
    }

    @Override
    public boolean supports(KernelCall call) {
        if (!supports(call.operation(), call.output().buffer().dtype())) {
            return false;
        }
        if (call.operation() == Operation.COPY) {
            KernelTensor source = call.inputs().getFirst();
            if (source.buffer() == call.output().buffer() && !source.layout().equals(call.output().layout())) {
                return false;
            }
            if (!source.buffer().dtype().isLowPrecision()) {
                return true;
            }
            return whole(source) && whole(call.output());
        }
        if (call.inputs().stream().anyMatch(t -> t.buffer().dtype() != DType.FP32)
                || !call.output().layout().isContiguous()) {
            return false;
        }
        if (call.operation() == Operation.MATMUL) {
            return call.output().layout().shape().rank() == 2
                    && call.inputs().stream().allMatch(t -> matrixLayout(t.layout()));
        }
        if (call.operation() == Operation.FILL) {
            return true;
        }
        if (mode == GpuMode.THROUGHPUT && call.operation() != Operation.SCALE && call.operation() != Operation.AXPY) {
            return call.output().layout().shape().rank() <= StridedView.MAX_RANK
                    && call.inputs().stream().allMatch(t -> t.layout().shape().rank() <= StridedView.MAX_RANK);
        }
        return call.inputs().stream().allMatch(t -> t.layout().isContiguous()
                && t.layout().shape().equals(call.output().layout().shape()));
    }

    @Override
    public Buffer allocate(DType dtype, long elements) {
        ensureOpen();
        if (elements < 0 || elements > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("GPU element count out of range");
        }
        long bytes = Math.max(4, Math.addExact(Math.multiplyExact(elements, width(dtype)), 3) & ~3L);
        var buffer = gpu.allocate(new BufferDescriptor(bytes,
                Set.of(BufferUsage.STORAGE, BufferUsage.TRANSFER_SOURCE, BufferUsage.TRANSFER_DESTINATION), Placement.PRIVATE, "nnjava"));
        var allocation = new Allocation(buffer, dtype, elements);
        allocations.add(allocation);
        return allocation;
    }

    @Override
    public Completion execute(KernelCall call) {
        ensureOpen();
        own(call.output().buffer());
        call.inputs().forEach(t -> own(t.buffer()));
        if (!supports(call)) {
            throw new UnsupportedOperationException("GPU " + mode + " does not support " + call.operation() + " for these types/layouts");
        }
        if (call.output().layout().shape().elements() == 0) {
            return Completion.completed();
        }
        return submit(encoder -> encode(encoder, call), () -> {
        });
    }

    private void encode(GpuCommandEncoder encoder, KernelCall call) {
        Operation op = call.operation();
        KernelTensor out = call.output();
        if (op == Operation.COPY) {
            copy(encoder, call.inputs().getFirst(), out);
            tiers.put(op, "DynamisGPU storage");
            return;
        }
        if (op == Operation.FILL) {
            try (var slice = range(out)) {
                encoder.fill(slice, Float.floatToRawIntBits(call.scalar()));
            }
            tiers.put(op, "DynamisGPU storage");
            return;
        }
        if (op == Operation.MATMUL) {
            matrixProduct(encoder, call);
            return;
        }
        KernelTensor a = call.inputs().getFirst();
        int n = Math.toIntExact(out.layout().shape().elements());
        GpuBuffer av = own(a.buffer()).buffer;
        GpuBuffer ov = own(out.buffer()).buffer;
        long ao = a.layout().offset(), oo = out.layout().offset();
        if (op == Operation.SCALE || op == Operation.AXPY) {
            if (blas == null) {
                blas = new GpuBlasf(gpu);
            }
            if (op == Operation.SCALE) {
                copy(encoder, a, out);
                blas.scal(encoder, call.scalar(), ov, oo, n);
            } else {
                blas.axpy(encoder, call.scalar(), av, ao, ov, oo, n);
            }
            tiers.put(op, "vectrix-gpu");
            return;
        }
        if (mode == GpuMode.THROUGHPUT) {
            if (elementwise == null) {
                elementwise = Elementwise.create(gpu);
            }
            var output = view(out.layout());
            var first = view(a.layout().broadcastTo(out.layout().shape()));
            if (call.inputs().size() == 2) {
                KernelTensor b = call.inputs().get(1);
                var binary = switch (op) {
                    case ADD -> Elementwise.Binary.ADD;
                    case SUBTRACT -> Elementwise.Binary.SUB;
                    case MULTIPLY -> Elementwise.Binary.MUL;
                    case DIVIDE -> Elementwise.Binary.DIV;
                    default -> throw new UnsupportedOperationException(op.toString());
                };
                elementwise.binary(encoder, binary, av, first, own(b.buffer()).buffer,
                        view(b.layout().broadcastTo(out.layout().shape())), ov, output);
            } else {
                elementwise.unary(encoder, Elementwise.Unary.valueOf(op.name()), av, first, ov, output);
            }
            tiers.put(op, "dynamis-gpu-primitives");
        } else {
            if (pointwise == null) {
                pointwise = new GpuPointwise(gpu);
            }
            if (call.inputs().size() == 2) {
                KernelTensor b = call.inputs().get(1);
                GpuBuffer bv = own(b.buffer()).buffer;
                long bo = b.layout().offset();
                switch (op) {
                    case ADD -> pointwise.add(encoder, av, ao, bv, bo, ov, oo, n);
                    case SUBTRACT -> pointwise.sub(encoder, av, ao, bv, bo, ov, oo, n);
                    case MULTIPLY -> pointwise.mul(encoder, av, ao, bv, bo, ov, oo, n);
                    case DIVIDE -> pointwise.div(encoder, av, ao, bv, bo, ov, oo, n);
                    default -> throw new UnsupportedOperationException(op.toString());
                }
            } else {
                switch (op) {
                    case EXP -> pointwise.exp(encoder, av, ao, ov, oo, n);
                    case LOG -> pointwise.log(encoder, av, ao, ov, oo, n);
                    case SQRT -> pointwise.sqrt(encoder, av, ao, ov, oo, n);
                    default -> throw new UnsupportedOperationException(op.toString());
                }
            }
            tiers.put(op, "vectrix-gpu");
        }
    }

    private void matrixProduct(GpuCommandEncoder encoder, KernelCall call) {
        KernelTensor a = call.inputs().get(0), b = call.inputs().get(1), out = call.output();
        int m = Math.toIntExact(a.layout().shape().dimension(0));
        int k = Math.toIntExact(a.layout().shape().dimension(1));
        int n = Math.toIntExact(b.layout().shape().dimension(1));
        if (k == 0) {
            try (var slice = range(out)) {
                encoder.fill(slice, 0);
            }
            return;
        }
        boolean ta = a.layout().stride(1) != 1, tb = b.layout().stride(1) != 1;
        int lda = Math.toIntExact(a.layout().stride(ta ? 1 : 0));
        int ldb = Math.toIntExact(b.layout().stride(tb ? 1 : 0));
        if (mode == GpuMode.EXACT) {
            if (exactGemm == null) {
                exactGemm = new GpuGemmf(gpu);
            }
            exactGemm.gemm(encoder, ta, tb, m, n, k, 1, own(a.buffer()).buffer, a.layout().offset(), lda,
                    own(b.buffer()).buffer, b.layout().offset(), ldb, 0, own(out.buffer()).buffer, out.layout().offset(), n);
            tiers.put(Operation.MATMUL, "vectrix-gpu");
        } else {
            if (throughputGemm == null) {
                throughputGemm = GemmF16.create(gpu, GemmF16.Input.FP32, Precision.FP32);
            }
            List<Allocation> temporary = new ArrayList<>();
            try {
                KernelTensor alignedA = zeroOffset(encoder, a, temporary, true);
                KernelTensor alignedB = zeroOffset(encoder, b, temporary, true);
                KernelTensor alignedOut = zeroOffset(encoder, out, temporary, false);
                boolean at = alignedA.layout().stride(1) != 1, bt = alignedB.layout().stride(1) != 1;
                int astride = Math.toIntExact(alignedA.layout().stride(at ? 1 : 0));
                int bstride = Math.toIntExact(alignedB.layout().stride(bt ? 1 : 0));
                var shape = new GemmF16.Shape(m, n, k, astride, bstride, n, 1, 0, bt, 1, 0, 0, 0, at);
                try (var av = tail(alignedA); var bv = tail(alignedB); var ov = tail(alignedOut)) {
                    throughputGemm.encode(encoder, av, bv, ov, shape);
                }
                if (alignedOut != out) {
                    copy(encoder, alignedOut, out);
                }
            } finally {
                // Recorded commands retain native storage independently of these handles.
                temporary.forEach(Allocation::close);
            }
            tiers.put(Operation.MATMUL, "dynamis-gpu-primitives");
        }
    }

    private KernelTensor zeroOffset(GpuCommandEncoder encoder, KernelTensor tensor,
                                    List<Allocation> temporary, boolean input) {
        if (tensor.layout().offset() == 0) {
            return tensor;
        }
        Allocation allocation = (Allocation) allocate(tensor.buffer().dtype(), tensor.layout().shape().elements());
        temporary.add(allocation);
        KernelTensor packed = new KernelTensor(allocation, TensorLayout.contiguous(tensor.layout().shape()));
        if (input) {
            copy(encoder, tensor, packed);
        }
        return packed;
    }

    private GpuBuffer range(KernelTensor tensor) {
        return own(tensor.buffer()).buffer.slice(tensor.layout().offset() * width(tensor.buffer().dtype()),
                tensor.layout().shape().elements() * width(tensor.buffer().dtype()));
    }

    private GpuBuffer tail(KernelTensor tensor) {
        var buffer = own(tensor.buffer()).buffer;
        long offset = tensor.layout().offset() * width(tensor.buffer().dtype());
        return buffer.slice(offset, buffer.bytes() - offset);
    }

    private void copy(GpuCommandEncoder encoder, KernelTensor source, KernelTensor target) {
        if (source.buffer() == target.buffer() && source.layout().equals(target.layout())) {
            return;
        }
        var a = own(source.buffer());
        var b = own(target.buffer());
        if (a.dtype.isLowPrecision()) {
            encoder.copy(a.buffer, b.buffer);
            return;
        }
        if (source.layout().isContiguous() && target.layout().isContiguous()) {
            try (var av = range(source); var bv = range(target)) {
                encoder.copy(av, bv);
            }
        } else {
            int bytes = width(a.dtype);
            for (long i = 0; i < target.layout().shape().elements(); i++) {
                try (var av = a.buffer.slice(source.layout().index(i) * bytes, bytes);
                     var bv = b.buffer.slice(target.layout().index(i) * bytes, bytes)) {
                    encoder.copy(av, bv);
                }
            }
        }
    }

    @Override
    public Completion upload(Buffer target, float[] source) {
        return upload(target, source, DType.FP32, source.length);
    }

    @Override
    public Completion upload(Buffer target, int[] source) {
        return upload(target, source, DType.INT32, source.length);
    }

    @Override
    public Completion upload(Buffer target, boolean[] source) {
        return upload(target, source, DType.BOOL, source.length);
    }

    @Override
    public Completion upload(Buffer target, short[] source) {
        if (!target.dtype().isLowPrecision()) {
            throw new IllegalArgumentException("Raw shorts require FP16/BF16");
        }
        return upload(target, source, target.dtype(), source.length);
    }

    private Completion upload(Buffer target, Object values, DType dtype, int count) {
        Allocation allocation = own(target);
        if (allocation.dtype != dtype || allocation.elements != count) {
            throw new IllegalArgumentException("Upload type/count mismatch");
        }
        if (count == 0) {
            return Completion.completed();
        }
        ByteBuffer bytes = ByteBuffer.allocate(Math.toIntExact(allocation.buffer.bytes())).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < count; i++) {
            switch (dtype) {
                case FP32 -> bytes.putFloat(((float[]) values)[i]);
                case INT32 -> bytes.putInt(((int[]) values)[i]);
                case BOOL -> bytes.putInt(((boolean[]) values)[i] ? 1 : 0);
                case FP16, BF16 -> bytes.putShort(((short[]) values)[i]);
            }
        }
        return submit(e -> e.upload(allocation.buffer, bytes.array()), () -> {
        });
    }

    @Override
    public Completion download(KernelTensor source, float[] target) {
        return download(source, target, DType.FP32, target.length);
    }

    @Override
    public Completion download(KernelTensor source, int[] target) {
        return download(source, target, DType.INT32, target.length);
    }

    @Override
    public Completion download(KernelTensor source, boolean[] target) {
        return download(source, target, DType.BOOL, target.length);
    }

    @Override
    public Completion download(KernelTensor source, short[] target) {
        if (!source.buffer().dtype().isLowPrecision()) {
            throw new IllegalArgumentException("Raw shorts require FP16/BF16");
        }
        return download(source, target, source.buffer().dtype(), target.length);
    }

    private Completion download(KernelTensor source, Object target, DType dtype, int count) {
        Allocation allocation = own(source.buffer());
        if (allocation.dtype != dtype || source.layout().shape().elements() != count) {
            throw new IllegalArgumentException("Download type/count mismatch");
        }
        if (count == 0) {
            return Completion.completed();
        }
        try (var encoder = gpu.encoder()) {
            var readback = encoder.download(allocation.buffer);
            return submitted(encoder, () -> {
                ByteBuffer bytes = ByteBuffer.wrap(readback.bytes()).order(ByteOrder.LITTLE_ENDIAN);
                for (int i = 0; i < count; i++) {
                    int index = Math.toIntExact(source.layout().index(i) * width(dtype));
                    switch (dtype) {
                        case FP32 -> ((float[]) target)[i] = bytes.getFloat(index);
                        case INT32 -> ((int[]) target)[i] = bytes.getInt(index);
                        case BOOL -> ((boolean[]) target)[i] = bytes.getInt(index) != 0;
                        case FP16, BF16 -> ((short[]) target)[i] = bytes.getShort(index);
                    }
                }
            });
        }
    }

    @Override
    public Completion randomNormal(Buffer target, long seed, long start, float deviation) {
        throw new UnsupportedOperationException("GPU normal initialization is not implemented yet; upload explicit weights");
    }

    private Completion submit(Consumer<GpuCommandEncoder> encode, Runnable publish) {
        ensureOpen();
        try (var encoder = gpu.encoder()) {
            encode.accept(encoder);
            return submitted(encoder, publish);
        }
    }

    private Completion submitted(GpuCommandEncoder encoder, Runnable publish) {
        for (var iterator = pending.iterator(); iterator.hasNext(); ) {
            var completion = iterator.next();
            if (completion.isDone()) {
                completion.await();
                iterator.remove();
            }
        }
        var completion = new GpuCompletion(gpu.queue().submit(encoder), publish);
        pending.add(completion);
        return completion;
    }

    private Allocation own(Buffer buffer) {
        ensureOpen();
        if (!(buffer instanceof Allocation a) || a.owner != this || a.isClosed()) {
            throw new IllegalArgumentException("Buffer is not owned by this live GPU backend");
        }
        return a;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("GPU backend is closed");
        }
    }

    @Override
    public void close() {
        if (closed) {
            return;
        }
        RuntimeException failure = null;
        for (var completion : pending) {
            try {
                completion.await();
            } catch (RuntimeException error) {
                if (failure == null) {
                    failure = error;
                } else if (failure != error) {
                    failure.addSuppressed(error);
                }
            }
        }
        pending.clear();
        for (var allocation : Set.copyOf(allocations)) {
            allocation.close();
        }
        for (AutoCloseable resource : new AutoCloseable[]{throughputGemm, elementwise, exactGemm, blas, pointwise, gpu}) {
            if (resource != null) {
                try {
                    resource.close();
                } catch (Exception error) {
                    if (failure == null) {
                        failure = new IllegalStateException("GPU close failed", error);
                    } else if (failure != error) {
                        failure.addSuppressed(error);
                    }
                }
            }
        }
        closed = true;
        if (failure != null) {
            throw failure;
        }
    }

    private final class Allocation implements Buffer {
        private final GpuBackend owner = GpuBackend.this;
        private final GpuBuffer buffer;
        private final DType dtype;
        private final long elements;
        private boolean released;

        Allocation(GpuBuffer buffer, DType dtype, long elements) {
            this.buffer = buffer;
            this.dtype = dtype;
            this.elements = elements;
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
            return released;
        }

        @Override
        public void close() {
            if (!released) {
                buffer.close();
                released = true;
                allocations.remove(this);
            }
        }
    }
}

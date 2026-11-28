package org.dynamisengine.nn.runtime;

import org.dynamisengine.nn.autograd.GradientTape;
import org.dynamisengine.nn.backend.*;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;
import org.dynamisengine.nn.tensor.TensorAccess;

import java.util.*;
import java.util.function.Supplier;

/**
 * A thread-confined owner of tensors and ordered backend work. close() drains work
 * and invalidates the context's remaining tensor handles. Backend ownership transfers here.
 */
public final class ExecutionContext implements AutoCloseable {
    private final Thread owner = Thread.currentThread();
    private final Backend backend;
    private final Set<Buffer> buffers = new LinkedHashSet<>();
    private final List<Pending> pending = new ArrayList<>();
    private GradientTape activeTape;
    private RuntimeException failure;
    private boolean closed;

    private ExecutionContext(Backend backend) {
        this.backend = Objects.requireNonNull(backend, "backend");
    }

    public static ExecutionContext cpu() {
        return open("vectrix", "cpu");
    }

    /**
     * Select a provider explicitly; only that provider opens a numerical backend.
     */
    public static ExecutionContext open(String providerId, String deviceId) {
        return create(BackendProviders.open(providerId, deviceId));
    }

    public static ExecutionContext create(Backend backend) {
        return new ExecutionContext(backend);
    }

    private static void checkCount(int count, Shape shape) {
        if (shape.elements() != count) {
            throw new IllegalArgumentException("Host data count does not match " + shape);
        }
    }

    public Device device() {
        ensureOpen();
        return backend.device();
    }

    public boolean supports(Operation operation, DType dtype) {
        ensureOpen();
        return backend.supports(operation, dtype);
    }

    public boolean supportsNormalInitialization() { ensureOpen(); return backend.supportsNormalInitialization(); }
    public long maxElements(DType dtype) { ensureOpen(); return backend.maxElements(dtype); }

    public void checkThread() {
        if (Thread.currentThread() != owner) {
            throw new IllegalStateException("Execution contexts are thread-confined");
        }
    }

    public void ensureOpen() {
        checkThread();
        if (closed) {
            throw new IllegalStateException("Execution context is closed");
        }
        if (failure != null) {
            throw new IllegalStateException("Execution context has failed", failure);
        }
    }

    /**
     * Allocate uninitialized storage. Prefer tensor(), zeros(), or full() for ordinary values.
     */
    public Tensor empty(DType dtype, Shape shape) {
        return allocate(dtype, shape, false);
    }

    private Tensor allocate(DType dtype, Shape shape, boolean requiresGrad) {
        ensureOpen();
        Objects.requireNonNull(dtype, "dtype");
        Objects.requireNonNull(shape, "shape");
        if (requiresGrad && dtype != DType.FP32) {
            throw new IllegalArgumentException("Only FP32 tensors support gradients");
        }
        // Validate metadata before allocating backend resources.
        org.dynamisengine.nn.tensor.TensorLayout.contiguous(shape);
        reap();
        Buffer buffer = backend.allocate(dtype, shape.elements());
        buffers.add(buffer);
        try {
            return TensorAccess.create(this, buffer, shape, requiresGrad);
        } catch (RuntimeException error) {
            releaseBuffer(buffer);
            throw error;
        }
    }

    public Tensor tensor(float[] values, Shape shape) {
        return tensor(values, shape, false);
    }

    public Tensor tensor(float[] values, Shape shape, boolean requiresGrad) {
        checkCount(values.length, shape);
        Tensor result = allocate(DType.FP32, shape, requiresGrad);
        float[] snapshot = values.clone();
        return initialize(result, () -> backend.upload(TensorAccess.kernel(result).buffer(), snapshot));
    }

    public Tensor tensor(int[] values, Shape shape) {
        checkCount(values.length, shape);
        Tensor result = allocate(DType.INT32, shape, false);
        int[] snapshot = values.clone();
        return initialize(result, () -> backend.upload(TensorAccess.kernel(result).buffer(), snapshot));
    }

    public Tensor tensor(boolean[] values, Shape shape) {
        checkCount(values.length, shape);
        Tensor result = allocate(DType.BOOL, shape, false);
        boolean[] snapshot = values.clone();
        return initialize(result, () -> backend.upload(TensorAccess.kernel(result).buffer(), snapshot));
    }

    public Tensor zeros(DType dtype, Shape shape) {
        return switch (dtype) {
            case FP32 -> full(shape, 0f);
            case INT32 -> tensor(new int[Math.toIntExact(shape.elements())], shape);
            case BOOL -> tensor(new boolean[Math.toIntExact(shape.elements())], shape);
            case FP16, BF16 -> tensor(new short[Math.toIntExact(shape.elements())], dtype, shape);
        };
    }

    /**
     * Creates FP16/BF16 storage from raw 16-bit values; does not enable gradients.
     */
    public Tensor tensor(short[] values, DType dtype, Shape shape) {
        if (!dtype.isLowPrecision()) {
            throw new IllegalArgumentException("Raw shorts require FP16 or BF16");
        }
        checkCount(values.length, shape);
        Tensor result = allocate(dtype, shape, false);
        short[] snapshot = values.clone();
        return initialize(result, () -> backend.upload(TensorAccess.kernel(result).buffer(), snapshot));
    }

    public Tensor full(Shape shape, float value) {
        return full(shape, value, false);
    }

    /** Constant FP32 storage, optionally a trainable leaf, initialized by the selected backend. */
    public Tensor full(Shape shape, float value, boolean requiresGrad) {
        Tensor result = allocate(DType.FP32, shape, requiresGrad);
        try {
            fill(result, value);
            return result;
        } catch (RuntimeException error) {
            result.close();
            throw error;
        }
    }

    public Tensor normal(Shape shape, long seed, float standardDeviation, boolean requiresGrad) {
        ensureOpen();
        if (!backend.supportsNormalInitialization()) {
            throw new UnsupportedOperationException(device() + " does not support normal initialization");
        }
        if (!Float.isFinite(standardDeviation) || standardDeviation < 0) {
            throw new IllegalArgumentException("Invalid normal standard deviation");
        }
        Tensor result = allocate(DType.FP32, shape, requiresGrad);
        return initialize(result, () -> backend.randomNormal(TensorAccess.kernel(result).buffer(), seed, 0,
                standardDeviation));
    }

    private Tensor initialize(Tensor tensor, Supplier<Completion> operation) {
        try {
            TensorAccess.beforeWrite(tensor);
            enqueue(operation, List.of(tensor));
            return tensor;
        } catch (RuntimeException error) {
            tensor.close();
            throw error;
        }
    }

    /**
     * Internal dispatch boundary shared by all Ops and future backend adapters.
     */
    public void submit(Operation operation, List<Tensor> inputs, Tensor output, float scalar) {
        submit(operation, inputs, output, scalar, KernelAttributes.EMPTY);
    }

    public void submit(Operation operation, List<Tensor> inputs, Tensor output, float scalar, KernelAttributes attributes) {
        ensureOpen();
        requireContext(output);
        for (Tensor input : inputs) {
            requireContext(input);
        }
        if (!backend.supports(operation, output.dtype())) {
            throw new UnsupportedOperationException(device() + " does not support " + operation + "/" + output.dtype());
        }
        KernelCall call = new KernelCall(operation, inputs.stream().map(TensorAccess::kernel).toList(),
                TensorAccess.kernel(output), scalar, attributes);
        if (!backend.supports(call)) {
            throw new UnsupportedOperationException(device() + " does not support " + operation
                    + " for the supplied types/layouts");
        }
        TensorAccess.beforeWrite(output, operation == Operation.COPY);
        List<Tensor> retained = new ArrayList<>(inputs);
        retained.add(output);
        enqueue(() -> backend.execute(call), retained);
    }

    public void fill(Tensor target, float value) {
        submit(Operation.FILL, List.of(), target, value);
    }

    public void assign(Tensor target, Tensor source) {
        if (activeTape != null) {
            throw new IllegalStateException("Assignment must run outside gradient recording");
        }
        submit(Operation.COPY, List.of(source), target, 0f);
    }

    /**
     * In-place target += scale * source. Intended for updates outside gradient recording.
     */
    public void axpy(Tensor target, Tensor source, float scale) {
        if (activeTape != null) {
            throw new IllegalStateException("Parameter updates must run outside a gradient tape");
        }
        submit(Operation.AXPY, List.of(source), target, scale);
    }

    public void update(OptimizerCall.Settings settings, Tensor parameter, Tensor gradient, List<Tensor> slots) {
        ensureOpen();
        if (activeTape != null) {
            throw new IllegalStateException("Optimizer updates require completed backward");
        }
        List<Tensor> values = new ArrayList<>(List.of(parameter, gradient));
        values.addAll(slots);
        values.forEach(this::requireContext);
        OptimizerCall call = new OptimizerCall(settings, TensorAccess.kernel(parameter), TensorAccess.kernel(gradient),
                slots.stream().map(TensorAccess::kernel).toList());
        TensorAccess.beforeWrite(parameter);
        slots.forEach(TensorAccess::beforeWrite);
        enqueue(() -> backend.update(call), values);
    }

    public void requireContext(Tensor tensor) {
        if (tensor.context() != this) {
            throw new IllegalArgumentException("Tensor belongs to a different execution context");
        }
    }

    private void enqueue(Supplier<Completion> operation, List<Tensor> values) {
        reap();
        List<Tensor> retained = values.stream().map(Tensor::retain).toList();
        try {
            Completion completion = Objects.requireNonNull(operation.get(), "backend completion");
            pending.add(new Pending(completion, retained));
        } catch (RuntimeException error) {
            retained.forEach(Tensor::close);
            fail(error);
            throw error;
        }
        reap();
    }

    private void reap() {
        for (var iterator = pending.iterator(); iterator.hasNext(); ) {
            Pending work = iterator.next();
            if (work.completion.isDone()) {
                try {
                    work.completion.await();
                } catch (RuntimeException error) {
                    fail(error);
                } finally {
                    work.retained.forEach(Tensor::close);
                    iterator.remove();
                }
            }
        }
        if (failure != null) {
            throw new IllegalStateException("Backend execution failed", failure);
        }
    }

    private void fail(RuntimeException error) {
        if (failure == null) {
            failure = error;
        } else if (failure != error) {
            failure.addSuppressed(error);
        }
    }

    public void synchronize() {
        checkThread();
        if (closed) {
            throw new IllegalStateException("Execution context is closed");
        }
        for (Pending work : pending) {
            try {
                work.completion.await();
            } catch (RuntimeException error) {
                fail(error);
            } finally {
                work.retained.forEach(Tensor::close);
            }
        }
        pending.clear();
        if (failure != null) {
            throw new IllegalStateException("Backend execution failed", failure);
        }
    }

    public float[] readFloats(Tensor tensor) {
        requireDtype(tensor, DType.FP32);
        float[] values = new float[Math.toIntExact(tensor.shape().elements())];
        KernelTensor source = TensorAccess.kernel(tensor);
        enqueue(() -> backend.download(source, values), List.of(tensor));
        synchronize();
        return values;
    }

    public int[] readInts(Tensor tensor) {
        requireDtype(tensor, DType.INT32);
        int[] values = new int[Math.toIntExact(tensor.shape().elements())];
        KernelTensor source = TensorAccess.kernel(tensor);
        enqueue(() -> backend.download(source, values), List.of(tensor));
        synchronize();
        return values;
    }

    public boolean[] readBooleans(Tensor tensor) {
        requireDtype(tensor, DType.BOOL);
        boolean[] values = new boolean[Math.toIntExact(tensor.shape().elements())];
        KernelTensor source = TensorAccess.kernel(tensor);
        enqueue(() -> backend.download(source, values), List.of(tensor));
        synchronize();
        return values;
    }

    public short[] readShorts(Tensor tensor) {
        ensureOpen();
        requireContext(tensor);
        if (!tensor.dtype().isLowPrecision()) {
            throw new IllegalArgumentException("Expected FP16 or BF16 storage");
        }
        short[] values = new short[Math.toIntExact(tensor.shape().elements())];
        KernelTensor source = TensorAccess.kernel(tensor);
        enqueue(() -> backend.download(source, values), List.of(tensor));
        synchronize();
        return values;
    }

    private void requireDtype(Tensor tensor, DType dtype) {
        ensureOpen();
        requireContext(tensor);
        if (tensor.dtype() != dtype) {
            throw new IllegalArgumentException("Expected " + dtype + ", got " + tensor.dtype());
        }
    }

    public GradientTape gradientTape() {
        ensureOpen();
        return new GradientTape(this);
    }

    public GradientTape activeTape() {
        ensureOpen();
        return activeTape;
    }

    /**
     * Internal tape registration. Nested tapes are not supported by the initial engine.
     */
    public void attachTape(GradientTape tape) {
        ensureOpen();
        if (activeTape != null) {
            throw new IllegalStateException("Nested gradient tapes are not supported");
        }
        activeTape = tape;
    }

    public void detachTape(GradientTape tape) {
        checkThread();
        if (activeTape == tape) {
            activeTape = null;
        }
    }

    /**
     * Number of owned allocations, excluding backend scratch. Useful for diagnostics.
     */
    public int liveAllocations() {
        checkThread();
        return buffers.size();
    }

    /**
     * Internal callback invoked when the last tensor/storage owner releases an allocation.
     */
    public void releaseBuffer(Buffer buffer) {
        checkThread();
        buffers.remove(buffer);
        buffer.close();
    }

    @Override
    public void close() {
        checkThread();
        if (closed) {
            return;
        }
        try {
            if (activeTape != null) {
                activeTape.close();
            }
            synchronize();
        } finally {
            for (Buffer buffer : Set.copyOf(buffers)) {
                buffer.close();
            }
            buffers.clear();
            closed = true;
            backend.close();
        }
    }

    private record Pending(Completion completion, List<Tensor> retained) {
    }
}

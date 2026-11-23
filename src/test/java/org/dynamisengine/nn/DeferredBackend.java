package org.dynamisengine.nn;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;
import org.dynamisengine.nn.backend.Backend;
import org.dynamisengine.nn.backend.Buffer;
import org.dynamisengine.nn.backend.Completion;
import org.dynamisengine.nn.backend.KernelCall;
import org.dynamisengine.nn.backend.KernelTensor;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.backend.vectrix.VectrixBackend;
import org.dynamisengine.nn.runtime.Device;
import org.dynamisengine.nn.tensor.DType;

/** A deterministic deferred command queue using real CPU kernels when work is awaited. */
final class DeferredBackend implements Backend {
    private final VectrixBackend cpu = new VectrixBackend();
    private final ArrayDeque<Job> queue = new ArrayDeque<>();
    final List<Buffer> allocations = new ArrayList<>();
    final List<KernelCall> calls = new ArrayList<>();
    boolean failNextKernel;
    boolean closed;
    private RuntimeException failure;

    @Override public Device device() { return cpu.device(); }
    @Override public boolean supports(Operation operation, DType dtype) { return cpu.supports(operation, dtype); }

    @Override
    public Buffer allocate(DType dtype, long elements) {
        Buffer buffer = cpu.allocate(dtype, elements);
        allocations.add(buffer);
        return buffer;
    }

    @Override
    public Completion execute(KernelCall call) {
        calls.add(call);
        boolean fail = failNextKernel;
        failNextKernel = false;
        return defer(() -> {
            if (fail) {
                throw new IllegalStateException("Injected asynchronous kernel failure");
            }
            return cpu.execute(call);
        });
    }

    private Completion defer(Supplier<Completion> action) {
        Job job = new Job(action);
        queue.add(job);
        return job;
    }

    @Override public Completion upload(Buffer target, float[] source) {
        return defer(() -> cpu.upload(target, source));
    }
    @Override public Completion upload(Buffer target, int[] source) {
        return defer(() -> cpu.upload(target, source));
    }
    @Override public Completion upload(Buffer target, boolean[] source) {
        return defer(() -> cpu.upload(target, source));
    }
    @Override public Completion download(KernelTensor source, float[] target) {
        return defer(() -> cpu.download(source, target));
    }
    @Override public Completion download(KernelTensor source, int[] target) {
        return defer(() -> cpu.download(source, target));
    }
    @Override public Completion download(KernelTensor source, boolean[] target) {
        return defer(() -> cpu.download(source, target));
    }
    @Override public Completion randomNormal(Buffer target, long seed, long start, float standardDeviation) {
        return defer(() -> cpu.randomNormal(target, seed, start, standardDeviation));
    }

    private final class Job implements Completion {
        private final Supplier<Completion> action;
        private boolean done;
        private RuntimeException error;

        private Job(Supplier<Completion> action) {
            this.action = action;
        }

        @Override public boolean isDone() { return done; }

        @Override
        public void await() {
            while (!done) {
                Job next = queue.removeFirst();
                try {
                    if (failure != null) {
                        throw failure;
                    }
                    next.action.get().await();
                } catch (RuntimeException caught) {
                    failure = caught;
                    next.error = caught;
                } finally {
                    next.done = true;
                }
            }
            if (error != null) {
                throw error;
            }
        }
    }

    @Override
    public void close() {
        if (!closed) {
            try {
                while (!queue.isEmpty()) {
                    try {
                        queue.getFirst().await();
                    } catch (RuntimeException ignored) {
                        // Remaining jobs are terminally failed before resources are released.
                    }
                }
            } finally {
                closed = true;
                cpu.close();
            }
        }
    }
}

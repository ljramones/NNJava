package org.dynamisengine.nn.backend;

import org.dynamisengine.nn.runtime.Device;
import org.dynamisengine.nn.tensor.DType;

/**
 * Ordered backend SPI. Uploads retain their source until completion; downloads retain
 * their destination. execute may enqueue work and must preserve call order.
 * Throwing at submission means no work was accepted; accepted work reports errors
 * through its Completion. Completion is terminal only when resource use has stopped.
 */
public interface Backend extends AutoCloseable {
    Device device();
    boolean supports(Operation operation, DType dtype);
    Buffer allocate(DType dtype, long elements);
    Completion execute(KernelCall call);
    Completion upload(Buffer target, float[] source);
    Completion upload(Buffer target, int[] source);
    Completion upload(Buffer target, boolean[] source);
    Completion download(KernelTensor source, float[] target);
    Completion download(KernelTensor source, int[] target);
    Completion download(KernelTensor source, boolean[] target);
    Completion randomNormal(Buffer target, long seed, long start, float standardDeviation);

    /** Drain pending work before releasing backend resources. Must be idempotent. */
    @Override
    void close();
}

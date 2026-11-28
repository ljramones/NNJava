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

    /**
     * Refines support using the actual input types and layouts before submission.
     */
    default boolean supports(KernelCall call) {
        return supports(call.operation(), call.output().buffer().dtype());
    }

    Buffer allocate(DType dtype, long elements);

    /** Allocation metadata limits, available before model construction. */
    default long maxElements(DType dtype) { return Long.MAX_VALUE; }

    /** Whether counter-based normal initialization is implemented on this provider. */
    default boolean supportsNormalInitialization() { return true; }

    /** Optional provider sampling. False by default, with no host sampling fallback. */
    default boolean supportsSampling(org.dynamisengine.nn.ops.SamplingOptions options, long rows, long vocabulary) { return false; }

    Completion execute(KernelCall call);

    default Completion update(OptimizerCall call) {
        throw new UnsupportedOperationException("Stateful optimizer kernels unavailable");
    }

    Completion upload(Buffer target, float[] source);

    Completion upload(Buffer target, int[] source);

    Completion upload(Buffer target, boolean[] source);

    /**
     * Raw IEEE FP16/BF16 storage bits, with no implicit numerical conversion.
     */
    default Completion upload(Buffer target, short[] source) {
        throw new UnsupportedOperationException("16-bit storage uploads unavailable");
    }

    Completion download(KernelTensor source, float[] target);

    Completion download(KernelTensor source, int[] target);

    Completion download(KernelTensor source, boolean[] target);

    default Completion download(KernelTensor source, short[] target) {
        throw new UnsupportedOperationException("16-bit storage downloads unavailable");
    }

    Completion randomNormal(Buffer target, long seed, long start, float standardDeviation);

    /**
     * Drain pending work before releasing backend resources. Must be idempotent.
     */
    @Override
    void close();
}

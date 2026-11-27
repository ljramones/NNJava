package org.dynamisengine.nn.training;

/**
 * Parameter update and gradient clearing, independent of the application's training loop.
 */
public interface Optimizer extends AutoCloseable {
    void step();

    void zeroGrad();

    default long steps() {
        throw new UnsupportedOperationException("Step state unavailable");
    }

    default OptimizerState state() {
        throw new UnsupportedOperationException("Optimizer persistence unavailable");
    }

    default void validateState(OptimizerState state) {
        throw new UnsupportedOperationException("Optimizer persistence unavailable");
    }

    default void restoreState(OptimizerState state) {
        throw new UnsupportedOperationException("Optimizer persistence unavailable");
    }

    @Override
    void close();
}

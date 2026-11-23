package org.dynamisengine.nn.training;

/** Parameter update and gradient clearing, independent of the application's training loop. */
public interface Optimizer extends AutoCloseable {
    void step();
    void zeroGrad();

    @Override
    void close();
}

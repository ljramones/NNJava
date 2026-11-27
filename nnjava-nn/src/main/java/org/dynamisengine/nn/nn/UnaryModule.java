package org.dynamisengine.nn.nn;

import org.dynamisengine.nn.tensor.Tensor;

/**
 * A composable single-input module; the caller owns the returned tensor.
 */
public interface UnaryModule extends Module {
    Tensor forward(Tensor input);
}

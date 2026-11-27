package org.dynamisengine.nn.autograd;

import org.dynamisengine.nn.tensor.Tensor;

import java.util.List;

/**
 * Return owned input-gradient handles, or null entries for untracked inputs.
 */
@FunctionalInterface
public interface BackwardRule {
    Tensor[] apply(Tensor upstream, List<Tensor> savedInputs);
}

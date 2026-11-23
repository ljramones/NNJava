package org.dynamisengine.nn.autograd;

import java.util.List;
import org.dynamisengine.nn.tensor.Tensor;

/** Return owned input-gradient handles, or null entries for untracked inputs. */
@FunctionalInterface
public interface BackwardRule {
    Tensor[] apply(Tensor upstream, List<Tensor> savedInputs);
}

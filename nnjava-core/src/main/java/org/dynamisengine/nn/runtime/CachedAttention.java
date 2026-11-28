package org.dynamisengine.nn.runtime;

import org.dynamisengine.nn.tensor.Tensor;

/** Session attention state; the returned attended tensor is owned by the caller. */
public interface CachedAttention {
    long length();
    Tensor attend(Tensor query, Tensor key, Tensor value, Tensor allowed);
}

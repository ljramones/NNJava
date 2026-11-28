package org.dynamisengine.nn.generation;

import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Tensor;

/** Borrowed model/context with owned inference cursors; logits and next-token inputs stay as tensors. */
public interface TensorAutoregressiveModel {
    ExecutionContext executionContext();
    int vocabularySize();
    int contextLength();
    Cursor openSession();

    interface Cursor extends AutoCloseable {
        int length();
        /** Borrows INT32 [1,time]; returns an owned FP32 [vocabulary] tensor. */
        Tensor prefill(Tensor tokens);
        /** Borrows INT32 [1,1], including a sampled device tensor; returns owned FP32 [vocabulary]. */
        Tensor decode(Tensor token);
        @Override void close();
    }
}

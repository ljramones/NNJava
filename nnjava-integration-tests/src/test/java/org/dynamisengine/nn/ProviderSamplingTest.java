package org.dynamisengine.nn;

import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.ops.SamplingOptions;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ProviderSamplingTest {
    @Test void cpuProviderDoesNotAdvertiseOrSilentlySubstituteOptionalDeviceSampling() {
        try (var context = ExecutionContext.cpu(); var logits = context.tensor(new float[]{1, 2, 3}, Shape.of(3))) {
            for (var dtype : DType.values()) { assertFalse(context.supports(Operation.SAMPLE, dtype)); }
            assertFalse(context.supportsSampling(SamplingOptions.greedy(), 1, 3)); int before = context.liveAllocations();
            assertThrows(UnsupportedOperationException.class, () -> Ops.sample(logits, SamplingOptions.greedy(), 0));
            assertEquals(before, context.liveAllocations());
        }
    }
}

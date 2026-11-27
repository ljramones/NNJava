package org.dynamisengine.nn;

import static org.junit.jupiter.api.Assertions.*;
import java.util.List;
import org.dynamisengine.nn.backend.*;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.*;
import org.junit.jupiter.api.Test;

class KernelContractsTest {
    @Test void invalidMaximumBackwardSeedIsRejectedBeforeExecution() {
        try (var context = ExecutionContext.cpu(); var input = context.full(Shape.of(2, 2), 1);
                var seed = context.full(Shape.of(), 1); var output = context.empty(DType.FP32, input.shape())) {
            var attributes = KernelAttributes.of("axes", List.of(1)).and("keep", false);
            assertThrows(IllegalArgumentException.class,
                    () -> context.submit(Operation.MAX_BACKWARD, List.of(input, seed), output, 0, attributes));
            try (var valid = context.full(Shape.of(), 3)) { assertEquals(3, valid.item()); }
        }
    }

    @Test void invalidRmsGradientIndexIsRejectedBeforeExecution() {
        try (var context = ExecutionContext.cpu(); var input = context.full(Shape.of(2, 2), 1);
                var scale = context.full(Shape.of(2), 1); var seed = context.full(input.shape(), 1);
                var output = context.empty(DType.FP32, input.shape())) {
            var attributes = KernelAttributes.of("rms", true).and("epsilon", 1e-5f).and("which", 2);
            assertThrows(IllegalArgumentException.class,
                    () -> context.submit(Operation.NORM_BACKWARD, List.of(input, scale, seed), output, 0, attributes));
            try (var valid = context.full(Shape.of(), 3)) { assertEquals(3, valid.item()); }
        }
    }
}

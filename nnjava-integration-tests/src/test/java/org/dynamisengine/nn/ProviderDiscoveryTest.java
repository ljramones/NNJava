package org.dynamisengine.nn;

import org.dynamisengine.nn.backend.BackendProviders;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ProviderDiscoveryTest {
    @Test
    void cpuStarterDiscoversTheVectrixProvider() {
        assertEquals(java.util.List.of("vectrix"), BackendProviders.available());
        try (var context = ExecutionContext.open("vectrix", "cpu");
             var value = context.full(Shape.of(2), 3)) {
            assertArrayEquals(new float[]{3, 3}, value.toFloatArray());
        }
        assertThrows(IllegalArgumentException.class, () -> ExecutionContext.open("vectrix", "missing"));
        assertThrows(IllegalStateException.class, () -> ExecutionContext.open("unavailable-gpu", "0"));
    }
}

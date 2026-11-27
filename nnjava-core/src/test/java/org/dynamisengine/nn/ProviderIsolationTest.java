package org.dynamisengine.nn;

import static org.junit.jupiter.api.Assertions.*;
import org.dynamisengine.nn.backend.BackendProviders;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.junit.jupiter.api.Test;

/** Core can be consumed without numerical implementations or native libraries. */
class ProviderIsolationTest {
    @Test void coreDoesNotAcquireAnImplicitCpuDependency() {
        assertTrue(BackendProviders.available().isEmpty());
        var error = assertThrows(IllegalStateException.class, ExecutionContext::cpu);
        assertTrue(error.getMessage().contains("vectrix"));
        assertTrue(error.getMessage().contains("backend artifact"));
    }
}

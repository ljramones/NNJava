package org.dynamisengine.nn.backend.vectrix;

import org.dynamisengine.nn.backend.Backend;
import org.dynamisengine.nn.backend.BackendProvider;
import org.dynamisengine.nn.runtime.Device;

import java.util.List;

/**
 * Vectrix 4.1 CPU provider. Discovery does not allocate numerical resources.
 */
public final class VectrixBackendProvider implements BackendProvider {
    @Override
    public String id() {
        return "vectrix";
    }

    @Override
    public List<Device> devices() {
        return List.of(new Device(id(), "cpu"));
    }

    @Override
    public Backend create(String deviceId) {
        if (!"cpu".equals(deviceId)) {
            throw new IllegalArgumentException("Unknown Vectrix CPU device: " + deviceId);
        }
        return new VectrixBackend();
    }
}

package org.dynamisengine.nn.backend.gpu;

import java.util.List;

import org.dynamisengine.nn.backend.Backend;
import org.dynamisengine.nn.backend.BackendProvider;
import org.dynamisengine.nn.runtime.Device;

public final class ThroughputGpuProvider implements BackendProvider {
    @Override
    public String id() {
        return "dynamis-gpu-throughput";
    }

    @Override
    public List<Device> devices() {
        return GpuProviders.devices(id());
    }

    @Override
    public Backend create(String deviceId) {
        return GpuProviders.create(deviceId, GpuMode.THROUGHPUT);
    }
}

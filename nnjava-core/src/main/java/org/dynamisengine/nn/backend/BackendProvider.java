package org.dynamisengine.nn.backend;

import org.dynamisengine.nn.runtime.Device;

import java.util.List;

/**
 * Service-provider entry point. Construction and id() must not initialize native
 * libraries or devices. Device discovery is explicit; create() transfers a new
 * backend's ownership to its caller.
 */
public interface BackendProvider {
    String id();

    List<Device> devices();

    Backend create(String deviceId);
}

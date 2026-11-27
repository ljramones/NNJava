package org.dynamisengine.nn.runtime;

import java.util.Objects;

/**
 * Backend identity and a device identifier within that backend.
 */
public record Device(String backend, String id) {
    public Device {
        Objects.requireNonNull(backend, "backend");
        Objects.requireNonNull(id, "id");
    }
}

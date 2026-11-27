package org.dynamisengine.nn.backend;

import java.util.List;
import java.util.Objects;
import java.util.ServiceLoader;

/**
 * Resolves providers without a dependency on any concrete numerical backend.
 */
public final class BackendProviders {
    private BackendProviders() {
    }

    /**
     * Lists provider IDs without probing hardware or initializing native runtimes.
     */
    public static List<String> available() {
        return ServiceLoader.load(BackendProvider.class).stream()
                .map(ServiceLoader.Provider::get).map(BackendProvider::id).sorted().toList();
    }

    public static Backend open(String providerId, String deviceId) {
        Objects.requireNonNull(providerId, "providerId");
        Objects.requireNonNull(deviceId, "deviceId");
        var matches = ServiceLoader.load(BackendProvider.class).stream()
                .map(ServiceLoader.Provider::get).filter(p -> providerId.equals(p.id())).toList();
        if (matches.isEmpty()) {
            throw new IllegalStateException("Numerical provider '" + providerId
                    + "' is unavailable. Add its NNJava backend artifact to the class/module path.");
        }
        if (matches.size() != 1) {
            throw new IllegalStateException("Multiple numerical providers registered as '" + providerId + "'");
        }
        return Objects.requireNonNull(matches.getFirst().create(deviceId), "provider backend");
    }
}

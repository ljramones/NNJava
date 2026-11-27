package org.dynamisengine.nn.state;

import java.util.Map;
import java.util.Objects;

/**
 * Portable component state with no dependency on training or archive codecs.
 */
public record ComponentState(String kind, Map<String, String> configuration,
                             Map<String, Long> counters, Map<String, TensorSnapshot> tensors) {
    public ComponentState {
        Objects.requireNonNull(kind, "kind");
        configuration = Map.copyOf(configuration);
        counters = Map.copyOf(counters);
        tensors = Map.copyOf(tensors);
    }
}

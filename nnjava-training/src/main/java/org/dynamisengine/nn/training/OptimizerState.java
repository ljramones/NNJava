package org.dynamisengine.nn.training;

import org.dynamisengine.nn.state.TensorSnapshot;

import java.util.Map;

public record OptimizerState(String kind, Map<String, String> configuration, Map<String, Long> counters,
                             Map<String, TensorSnapshot> tensors) {
    public OptimizerState {
        configuration = Map.copyOf(configuration);
        counters = Map.copyOf(counters);
        tensors = Map.copyOf(tensors);
    }
}

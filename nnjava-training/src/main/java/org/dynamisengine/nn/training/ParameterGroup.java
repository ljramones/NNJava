package org.dynamisengine.nn.training;

import org.dynamisengine.nn.nn.Parameter;

import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Ordered named parameters with an independent base learning rate and decay.
 */
public record ParameterGroup(Map<String, Parameter> parameters, float learningRate, float weightDecay) {
    public ParameterGroup {
        parameters = Collections.unmodifiableMap(new LinkedHashMap<>(parameters));
        if (parameters.isEmpty() || !Float.isFinite(learningRate) || learningRate < 0 || !Float.isFinite(weightDecay) || weightDecay < 0
                || parameters.entrySet().stream().anyMatch(e -> e.getKey() == null || e.getKey().isBlank() || e.getValue() == null)) {
            throw new IllegalArgumentException("Invalid parameter group");
        }
    }

    public static ParameterGroup of(Collection<Parameter> parameters, float learningRate, float decay) {
        Map<String, Parameter> named = new LinkedHashMap<>();
        int index = 0;
        for (Parameter p : parameters) {
            named.put(index++ + ":" + p.name(), p);
        }
        return new ParameterGroup(named, learningRate, decay);
    }
}

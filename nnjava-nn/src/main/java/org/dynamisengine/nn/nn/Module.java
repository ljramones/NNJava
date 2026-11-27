package org.dynamisengine.nn.nn;

import java.util.Map;

/**
 * Base contract for user-defined modules. Implementations own their registered parameters.
 */
public interface Module extends AutoCloseable {
    Map<String, Parameter> namedParameters();

    default Map<String, Module> children() {
        return Map.of();
    }

    default Map<String, String> configuration() {
        return Map.of("$type", getClass().getName());
    }

    default Map<String, Long> trainingState() {
        return Map.of("$training", isTraining() ? 1L : 0L);
    }

    default void validateTrainingState(Map<String, Long> state) {
        if (!state.keySet().equals(trainingState().keySet()) || state.get("$training") < 0 || state.get("$training") > 1) {
            throw new IllegalArgumentException("Incompatible module training state");
        }
    }

    default void restoreTrainingState(Map<String, Long> state) {
        validateTrainingState(state);
        training(state.get("$training") == 1);
    }

    boolean isTraining();

    void training(boolean training);

    @Override
    void close();
}

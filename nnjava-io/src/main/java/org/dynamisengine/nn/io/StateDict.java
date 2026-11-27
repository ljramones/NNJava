package org.dynamisengine.nn.io;

import org.dynamisengine.nn.nn.Module;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.state.TensorSnapshot;
import org.dynamisengine.nn.tensor.Tensor;

import java.util.*;

/**
 * Named weights, tied-weight aliases, and module execution/random state.
 */
public record StateDict(Map<String, TensorSnapshot> tensors, Map<String, String> aliases,
                        Map<String, Long> trainingState,
                        Map<String, String> configuration) {
    public StateDict {
        tensors = Map.copyOf(tensors);
        aliases = Map.copyOf(aliases);
        trainingState = Map.copyOf(trainingState);
        configuration = Map.copyOf(configuration);
        for (var alias : aliases.entrySet()) {
            if (tensors.containsKey(alias.getKey()) || !tensors.containsKey(alias.getValue())) {
                throw new IllegalArgumentException("Invalid state alias");
            }
        }
    }

    public static StateDict capture(Module module) {
        Map<String, TensorSnapshot> snapshots = new LinkedHashMap<>();
        Map<String, String> aliases = new LinkedHashMap<>();
        Map<Long, String> owners = new HashMap<>();
        for (var entry : module.namedParameters().entrySet()) {
            try (Tensor value = entry.getValue().value()) {
                if (value.context().activeTape() != null) {
                    throw new IllegalStateException("Snapshot after backward or outside recording");
                }
                String previous = owners.putIfAbsent(value.id(), entry.getKey());
                if (previous == null) {
                    snapshots.put(entry.getKey(), TensorSnapshot.capture(value));
                } else {
                    aliases.put(entry.getKey(), previous);
                }
            }
        }
        return new StateDict(snapshots, aliases, module.trainingState(), module.configuration());
    }

    public void validate(Module module) {
        if (!configuration.equals(module.configuration())) {
            throw new IllegalArgumentException("Module configuration mismatch");
        }
        var parameters = module.namedParameters();
        Set<String> names = new HashSet<>(tensors.keySet());
        names.addAll(aliases.keySet());
        if (!names.equals(parameters.keySet())) {
            throw new IllegalArgumentException("Model parameter names mismatch");
        }
        Map<String, Long> owners = new HashMap<>();
        Map<Long, String> canonicalOwners = new HashMap<>();
        for (var entry : parameters.entrySet()) {
            try (Tensor value = entry.getValue().value()) {
                if (value.context().activeTape() != null) {
                    throw new IllegalStateException("Restore requires no active tape");
                }
                String canonical = aliases.getOrDefault(entry.getKey(), entry.getKey());
                Long previous = owners.putIfAbsent(canonical, value.id());
                String previousCanonical = canonicalOwners.putIfAbsent(value.id(), canonical);
                if (previous != null && previous != value.id() || previousCanonical != null && !previousCanonical.equals(canonical)) {
                    throw new IllegalArgumentException("Tied parameter layout mismatch");
                }
                TensorSnapshot snapshot = tensors.get(canonical);
                if (snapshot == null) {
                    throw new IllegalArgumentException("Missing model weight " + canonical);
                }
                snapshot.validate(value);
            }
        }
        module.validateTrainingState(trainingState);
    }

    public void applyTo(Module module) {
        validate(module);
        Set<ExecutionContext> contexts = new HashSet<>();
        for (var entry : tensors.entrySet()) {
            try (Tensor target = module.namedParameters().get(entry.getKey()).value()) {
                entry.getValue().assignTo(target);
                contexts.add(target.context());
                target.clearGradient();
            }
        }
        contexts.forEach(ExecutionContext::synchronize);
        module.restoreTrainingState(trainingState);
    }
}

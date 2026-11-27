package org.dynamisengine.nn.training;

import org.dynamisengine.nn.io.StateArchive;
import org.dynamisengine.nn.io.StateDict;
import org.dynamisengine.nn.nn.Module;
import org.dynamisengine.nn.state.ComponentState;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

/**
 * Full training checkpoints composed from backend-neutral model archives.
 */
public final class Checkpoint {
    private Checkpoint() {
    }

    public static void save(Path path, Module model, Optimizer optimizer, Map<String, String> metadata) throws IOException {
        write(path, new Loaded(StateDict.capture(model), optimizer == null ? null : optimizer.state(), metadata));
    }

    public static void saveWeights(Path path, Module model) throws IOException {
        StateArchive.saveWeights(path, model);
    }

    public static void write(Path path, Loaded checkpoint) throws IOException {
        OptimizerState state = checkpoint.optimizer();
        ComponentState auxiliary = state == null ? null : new ComponentState(state.kind(), state.configuration(), state.counters(), state.tensors());
        StateArchive.write(path, new StateArchive.Loaded(checkpoint.model(), auxiliary, checkpoint.metadata()));
    }

    public static Loaded read(Path path) throws IOException {
        var archive = StateArchive.read(path);
        var state = archive.auxiliary();
        OptimizerState optimizer = state == null ? null : new OptimizerState(state.kind(), state.configuration(), state.counters(), state.tensors());
        return new Loaded(archive.model(), optimizer, archive.metadata());
    }

    public record Loaded(StateDict model, OptimizerState optimizer, Map<String, String> metadata) {
        public Loaded {
            metadata = Map.copyOf(metadata);
        }

        public void restore(Module target, Optimizer trainingOptimizer) {
            model.validate(target);
            if ((optimizer == null) != (trainingOptimizer == null)) {
                throw new IllegalArgumentException("Optimizer state presence mismatch");
            }
            if (trainingOptimizer != null) {
                trainingOptimizer.validateState(optimizer);
            }
            model.applyTo(target);
            if (trainingOptimizer != null) {
                trainingOptimizer.restoreState(optimizer);
            }
        }

        public void restoreModel(Module target) {
            model.applyTo(target);
        }
    }
}

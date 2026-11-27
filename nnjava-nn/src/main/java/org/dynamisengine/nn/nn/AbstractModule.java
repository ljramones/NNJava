package org.dynamisengine.nn.nn;

import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Recursive, ordered registration and ownership for custom modules.
 */
public abstract class AbstractModule implements Module {
    protected final ExecutionContext context;
    private final Map<String, Parameter> own = new LinkedHashMap<>();
    private final Map<String, Module> children = new LinkedHashMap<>();
    private boolean training = true;
    private boolean closed;

    protected AbstractModule(ExecutionContext context) {
        this.context = java.util.Objects.requireNonNull(context);
    }

    private static boolean contains(Module root, Module target) {
        for (Module child : root.children().values()) {
            if (child == target || contains(child, target)) {
                return true;
            }
        }
        return false;
    }

    protected final Parameter register(String name, Tensor value) {
        return register(name, new Parameter(name, value));
    }

    protected final Parameter register(String name, Parameter parameter) {
        checkName(name);
        if (own.containsKey(name) || children.containsKey(name)) {
            throw new IllegalArgumentException("Duplicate module name " + name);
        }
        try (Tensor value = parameter.value()) {
            context.requireContext(value);
        }
        own.put(name, parameter);
        return parameter;
    }

    protected final <T extends Module> T register(String name, T child) {
        checkName(name);
        if (child == this || contains(child, this) || own.containsKey(name) || children.containsKey(name)) {
            throw new IllegalArgumentException("Duplicate name or module registration cycle");
        }
        for (Parameter parameter : child.namedParameters().values()) {
            try (Tensor value = parameter.value()) {
                context.requireContext(value);
            }
        }
        children.put(name, child);
        child.training(training);
        return child;
    }

    private void checkName(String name) {
        ensureOpen();
        if (name == null || name.isBlank() || name.contains(".") || name.startsWith("$")) {
            throw new IllegalArgumentException("Invalid registration name");
        }
    }

    protected final Parameter constantParameter(String name, Shape shape, float value) {
        float[] values = new float[Math.toIntExact(shape.elements())];
        java.util.Arrays.fill(values, value);
        try (Tensor tensor = context.tensor(values, shape, true)) {
            return register(name, tensor);
        }
    }

    @Override
    public final Map<String, Parameter> namedParameters() {
        ensureOpen();
        Map<String, Parameter> result = new LinkedHashMap<>(own);
        children.forEach((prefix, child) -> child.namedParameters().forEach((name, parameter) -> result.put(prefix + "." + name, parameter)));
        return Collections.unmodifiableMap(result);
    }

    @Override
    public final Map<String, Module> children() {
        ensureOpen();
        return Collections.unmodifiableMap(children);
    }

    @Override
    public final boolean isTraining() {
        ensureOpen();
        return training;
    }

    @Override
    public void training(boolean training) {
        ensureOpen();
        this.training = training;
        children.values().forEach(child -> child.training(training));
    }

    protected Map<String, Long> localState() {
        return Map.of();
    }

    protected Map<String, String> localConfiguration() {
        return Map.of();
    }

    @Override
    public final Map<String, String> configuration() {
        ensureOpen();
        Map<String, String> values = new LinkedHashMap<>(localConfiguration());
        values.put("$type", getClass().getName());
        children.forEach((prefix, child) -> child.configuration().forEach((name, value) -> values.put(prefix + "." + name, value)));
        return Collections.unmodifiableMap(values);
    }

    protected void validateLocalState(Map<String, Long> state) {
    }

    protected void restoreLocalState(Map<String, Long> state) {
    }

    @Override
    public final Map<String, Long> trainingState() {
        ensureOpen();
        Map<String, Long> state = new LinkedHashMap<>(localState());
        state.put("$training", training ? 1L : 0L);
        children.forEach((prefix, child) -> child.trainingState().forEach((name, value) -> state.put(prefix + "." + name, value)));
        return Collections.unmodifiableMap(state);
    }

    private Map<String, Long> childState(Map<String, Long> state, String prefix) {
        Map<String, Long> result = new LinkedHashMap<>();
        state.forEach((key, value) -> {
            if (key.startsWith(prefix + ".")) {
                result.put(key.substring(prefix.length() + 1), value);
            }
        });
        return result;
    }

    @Override
    public final void validateTrainingState(Map<String, Long> state) {
        Module.super.validateTrainingState(state);
        validateLocalState(state);
        children.forEach((name, child) -> child.validateTrainingState(childState(state, name)));
    }

    @Override
    public final void restoreTrainingState(Map<String, Long> state) {
        validateTrainingState(state);
        training = state.get("$training") == 1;
        restoreLocalState(state);
        children.forEach((name, child) -> child.restoreTrainingState(childState(state, name)));
    }

    protected final void ensureOpen() {
        context.ensureOpen();
        if (closed) {
            throw new IllegalStateException("Module is closed");
        }
    }

    @Override
    public void close() {
        context.checkThread();
        if (!closed) {
            children.values().forEach(Module::close);
            own.values().forEach(Parameter::close);
            closed = true;
        }
    }
}

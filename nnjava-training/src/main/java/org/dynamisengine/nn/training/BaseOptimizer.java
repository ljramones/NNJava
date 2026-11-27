package org.dynamisengine.nn.training;

import org.dynamisengine.nn.backend.OptimizerCall;
import org.dynamisengine.nn.nn.Parameter;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.state.TensorSnapshot;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Tensor;

import java.util.*;

/**
 * Ordered, deduplicated groups and checkpointable slots for CPU optimizers.
 */
public abstract class BaseOptimizer implements Optimizer {
    protected final ExecutionContext context;
    protected final List<ParameterGroup> groups;
    private final List<Entry> entries = new ArrayList<>();
    private final LearningRateSchedule schedule;
    private final String kind;
    private final Map<String, String> configuration;
    private long steps;
    private boolean closed;

    protected BaseOptimizer(String kind, List<ParameterGroup> groups, LearningRateSchedule schedule,
                            Map<String, String> options, List<String> slotNames) {
        if (groups.isEmpty()) {
            throw new IllegalArgumentException("Optimizer needs parameter groups");
        }
        this.kind = kind;
        this.groups = List.copyOf(groups);
        this.schedule = Objects.requireNonNull(schedule);
        Map<String, String> config = new LinkedHashMap<>(options);
        config.put("schedule", schedule.id());
        ExecutionContext selected = null;
        Map<Long, Integer> seen = new HashMap<>();
        try {
            for (int group = 0; group < groups.size(); group++) {
                ParameterGroup current = groups.get(group);
                config.put("group." + group + ".lr", Float.toHexString(current.learningRate()));
                config.put("group." + group + ".decay", Float.toHexString(current.weightDecay()));
                for (var named : current.parameters().entrySet()) {
                    try (Tensor value = named.getValue().value()) {
                        if (selected == null) {
                            selected = value.context();
                        }
                        selected.requireContext(value);
                        Integer previous = seen.putIfAbsent(value.id(), group);
                        if (previous != null) {
                            if (previous != group) {
                                throw new IllegalArgumentException("Shared parameter occurs in different optimizer groups");
                            }
                            continue;
                        }
                        String key = "group." + group + "." + named.getKey();
                        config.put("parameter." + key, value.dtype() + ":" + value.shape());
                        Entry entry = new Entry(key, named.getValue(), group);
                        entries.add(entry);
                        for (String slot : slotNames) {
                            entry.slots.put(slot, selected.zeros(DType.FP32, value.shape()));
                        }
                    }
                }
            }
        } catch (RuntimeException error) {
            entries.forEach(e -> e.slots.values().forEach(Tensor::close));
            throw error;
        }
        if (entries.isEmpty()) {
            throw new IllegalArgumentException("No unique parameters");
        }
        context = selected;
        configuration = Collections.unmodifiableMap(config);
    }

    protected abstract OptimizerCall.Settings settings(float rate, float decay, long step);

    @Override
    public final void step() {
        ensureOpen();
        if (context.activeTape() != null) {
            throw new IllegalStateException("Finish backward before updating parameters");
        }
        context.synchronize();
        long next = Math.addExact(steps, 1);
        float[] rates = new float[groups.size()];
        for (int i = 0; i < rates.length; i++) {
            rates[i] = schedule.rate(next, groups.get(i).learningRate());
            if (!Float.isFinite(rates[i]) || rates[i] < 0) {
                throw new IllegalArgumentException("Schedule returned an invalid rate");
            }
        }
        List<Update> updates = new ArrayList<>();
        try {
            for (Entry entry : entries) {
                Tensor parameter = entry.parameter.value();
                if (!parameter.hasGradient()) {
                    parameter.close();
                    continue;
                }
                Tensor gradient = parameter.gradient();
                Update update = new Update(entry, parameter, gradient);
                updates.add(update);
                if (!parameter.shape().equals(gradient.shape()) || gradient.dtype() != DType.FP32) {
                    throw new IllegalArgumentException("Invalid parameter gradient");
                }
                Gradients.requireFinite(gradient);
                settings(rates[entry.group], groups.get(entry.group).weightDecay(), Math.addExact(entry.updates, 1));
            }
            for (Update update : updates) {
                Entry entry = update.entry;
                context.update(settings(rates[entry.group], groups.get(entry.group).weightDecay(), entry.updates + 1),
                        update.parameter, update.gradient, List.copyOf(entry.slots.values()));
            }
            context.synchronize();
            for (Update update : updates) {
                update.entry.updates++;
            }
            steps = next;
        } finally {
            updates.forEach(Update::close);
        }
    }

    @Override
    public final void zeroGrad() {
        ensureOpen();
        entries.forEach(e -> e.parameter.clearGradient());
    }

    @Override
    public final long steps() {
        ensureOpen();
        return steps;
    }

    @Override
    public final OptimizerState state() {
        ensureOpen();
        context.synchronize();
        Map<String, Long> counters = new LinkedHashMap<>();
        counters.put("steps", steps);
        Map<String, TensorSnapshot> tensors = new LinkedHashMap<>();
        for (Entry entry : entries) {
            counters.put(entry.key, entry.updates);
            entry.slots.forEach((name, value) -> tensors.put(entry.key + "." + name, TensorSnapshot.capture(value)));
        }
        return new OptimizerState(kind, configuration, counters, tensors);
    }

    @Override
    public final void validateState(OptimizerState state) {
        ensureOpen();
        if (context.activeTape() != null) {
            throw new IllegalStateException("Optimizer restore requires no active tape");
        }
        if (!state.kind().equals(kind) || !state.configuration().equals(configuration)) {
            throw new IllegalArgumentException("Optimizer kind, groups, schedule, or configuration mismatch");
        }
        Set<String> counters = new HashSet<>(Set.of("steps"));
        Set<String> slots = new HashSet<>();
        for (Entry entry : entries) {
            counters.add(entry.key);
            entry.slots.forEach((name, value) -> {
                String key = entry.key + "." + name;
                slots.add(key);
                TensorSnapshot snapshot = state.tensors().get(key);
                if (snapshot == null) {
                    throw new IllegalArgumentException("Missing optimizer slot " + key);
                }
                snapshot.validate(value);
            });
        }
        if (!state.counters().keySet().equals(counters) || !state.tensors().keySet().equals(slots)
                || state.counters().values().stream().anyMatch(v -> v < 0)
                || state.counters().entrySet().stream().anyMatch(e -> !e.getKey().equals("steps") && e.getValue() > state.counters().get("steps"))) {
            throw new IllegalArgumentException("Optimizer counters or slots mismatch");
        }
    }

    @Override
    public final void restoreState(OptimizerState state) {
        validateState(state);
        for (Entry entry : entries) {
            entry.slots.forEach((name, value) -> state.tensors().get(entry.key + "." + name).assignTo(value));
        }
        context.synchronize();
        steps = state.counters().get("steps");
        for (Entry entry : entries) {
            entry.updates = state.counters().get(entry.key);
        }
    }

    private void ensureOpen() {
        context.ensureOpen();
        if (closed) {
            throw new IllegalStateException("Optimizer is closed");
        }
    }

    @Override
    public final void close() {
        if (!closed) {
            entries.forEach(e -> e.slots.values().forEach(Tensor::close));
            closed = true;
        }
    }

    private static final class Entry {
        final String key;
        final Parameter parameter;
        final int group;
        final Map<String, Tensor> slots = new LinkedHashMap<>();
        long updates;

        Entry(String key, Parameter parameter, int group) {
            this.key = key;
            this.parameter = parameter;
            this.group = group;
        }
    }

    private record Update(Entry entry, Tensor parameter, Tensor gradient) implements AutoCloseable {
        @Override
        public void close() {
            parameter.close();
            gradient.close();
        }
    }
}

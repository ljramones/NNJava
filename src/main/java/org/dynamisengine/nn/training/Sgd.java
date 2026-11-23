package org.dynamisengine.nn.training;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.dynamisengine.nn.nn.Parameter;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Tensor;

/** Plain SGD. Shared parameters are updated once per step, by tensor identity. */
public final class Sgd implements Optimizer {
    private final List<Parameter> parameters;
    private final ExecutionContext context;
    private final float learningRate;
    private boolean closed;

    public Sgd(Collection<Parameter> parameters, float learningRate) {
        if (!Float.isFinite(learningRate) || learningRate < 0 || parameters.isEmpty()) {
            throw new IllegalArgumentException("SGD requires parameters and a finite nonnegative learning rate");
        }
        Set<Long> identities = new HashSet<>();
        List<Parameter> unique = new ArrayList<>();
        ExecutionContext selected = null;
        for (Parameter parameter : parameters) {
            try (Tensor value = parameter.value()) {
                if (selected == null) {
                    selected = value.context();
                }
                selected.requireContext(value);
                if (identities.add(value.id())) {
                    unique.add(parameter);
                }
            }
        }
        this.parameters = List.copyOf(unique);
        this.context = selected;
        this.learningRate = learningRate;
    }

    @Override
    public void step() {
        ensureOpen();
        if (context.activeTape() != null) {
            throw new IllegalStateException("Finish backward before updating parameters");
        }
        context.synchronize();
        for (Parameter parameter : parameters) {
            try (Tensor value = parameter.value()) {
                if (value.hasGradient()) {
                    try (Tensor gradient = value.gradient()) {
                        context.axpy(value, gradient, -learningRate);
                    }
                }
            }
        }
    }

    @Override
    public void zeroGrad() {
        ensureOpen();
        parameters.forEach(Parameter::clearGradient);
    }

    private void ensureOpen() {
        context.ensureOpen();
        if (closed) {
            throw new IllegalStateException("Optimizer is closed");
        }
    }

    @Override
    public void close() {
        closed = true;
    }
}

package org.dynamisengine.nn.nn;

import org.dynamisengine.nn.model.ComponentSpec;
import org.dynamisengine.nn.runtime.ExecutionContext;

/** Reusable configuration; validation and specification inspection allocate no model tensors. */
public interface ModelDefinition<M extends Module> {
    String name();
    ComponentSpec specification();
    void validate(ExecutionContext context);
    M build(ExecutionContext context);
    /** Definitions explicitly opt into alternative initialization; custom definitions retain their existing contract. */
    default void validate(ExecutionContext context, ParameterInitialization initialization) {
        if (java.util.Objects.requireNonNull(initialization) != ParameterInitialization.DEFAULT) {
            throw new UnsupportedOperationException(name() + ": alternative parameter initialization is unsupported");
        }
        validate(context);
    }
    default M build(ExecutionContext context, ParameterInitialization initialization) {
        if (java.util.Objects.requireNonNull(initialization) != ParameterInitialization.DEFAULT) {
            throw new UnsupportedOperationException(name() + ": alternative parameter initialization is unsupported");
        }
        return build(context);
    }
}

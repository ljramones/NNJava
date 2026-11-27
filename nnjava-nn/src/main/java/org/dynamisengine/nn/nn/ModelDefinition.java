package org.dynamisengine.nn.nn;

import org.dynamisengine.nn.model.ComponentSpec;
import org.dynamisengine.nn.runtime.ExecutionContext;

/** Reusable configuration; validation and specification inspection allocate no model tensors. */
public interface ModelDefinition<M extends Module> {
    String name();
    ComponentSpec specification();
    void validate(ExecutionContext context);
    M build(ExecutionContext context);
}

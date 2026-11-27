package org.dynamisengine.nn.nn;

import org.dynamisengine.nn.model.ComponentSpec;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;

/** Immutable layer specification. Custom factories own cleanup if construction fails. */
public interface LayerDefinition {
    Shape outputShape(Shape inputShape);
    void validate(ExecutionContext context, Shape inputShape, String path);
    UnaryModule build(ExecutionContext context, Shape inputShape, long seed);
    ComponentSpec specification(String name);
}

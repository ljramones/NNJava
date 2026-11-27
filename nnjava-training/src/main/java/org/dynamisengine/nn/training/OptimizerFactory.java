package org.dynamisengine.nn.training;

import java.util.Map;
import org.dynamisengine.nn.nn.Parameter;

/** Creates an owned optimizer while borrowing the model's named parameters. */
@FunctionalInterface
public interface OptimizerFactory {
    Optimizer build(Map<String, Parameter> parameters);
}

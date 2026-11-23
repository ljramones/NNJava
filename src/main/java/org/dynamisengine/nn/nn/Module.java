package org.dynamisengine.nn.nn;

import java.util.Map;

/** Base contract for user-defined modules. Implementations own their registered parameters. */
public interface Module extends AutoCloseable {
    Map<String, Parameter> namedParameters();
    boolean isTraining();
    void training(boolean training);

    @Override
    void close();
}

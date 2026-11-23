package org.dynamisengine.nn.backend;

/** Initial portable operation identifiers. View operations need no kernel dispatch. */
public enum Operation {
    COPY, FILL, ADD, SUBTRACT, MULTIPLY, SCALE, SUM, SUM_TO, MATMUL, AXPY
}

package org.dynamisengine.nn.backend.gpu;

/**
 * Numerical selection; unsupported operations never switch to CPU execution.
 */
public enum GpuMode {EXACT, THROUGHPUT}

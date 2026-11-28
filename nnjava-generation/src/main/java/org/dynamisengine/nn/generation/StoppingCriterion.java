package org.dynamisengine.nn.generation;

/** Called after a generated token has been committed to the cache and history. */
@FunctionalInterface
public interface StoppingCriterion {
    boolean shouldStop(TokenHistory history);
}

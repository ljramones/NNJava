package org.dynamisengine.nn.generation;

/** Synchronous hook operating on an owned working copy; negative infinity bans a token. */
@FunctionalInterface
public interface LogitsProcessor {
    void process(TokenHistory history,float[] logits);
}

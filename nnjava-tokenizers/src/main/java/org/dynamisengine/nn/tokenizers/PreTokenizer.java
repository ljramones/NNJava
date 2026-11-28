package org.dynamisengine.nn.tokenizers;

import java.util.List;

/**
 * Must return nonempty pieces covering the normalized input exactly, in order.
 */
@FunctionalInterface
public interface PreTokenizer {
    List<String> split(String text);
}

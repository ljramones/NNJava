package org.dynamisengine.nn.tokenizers;

import java.util.Map;
import java.util.Set;

/**
 * Minimal complete UTF-8 byte vocabulary; no pretrained-model compatibility is implied.
 */
public final class ByteTokenizerProvider implements TokenizerProvider {
    public String id() {
        return "utf8-bytes";
    }

    public Tokenizer create(Map<String, String> configuration) {
        if (!Set.of("normalization").containsAll(configuration.keySet())) {
            throw new IllegalArgumentException("Unknown utf8-bytes configuration setting");
        }
        return BytePairTokenizer.builder().normalization(TextNormalization.valueOf(configuration.getOrDefault("normalization", "NONE"))).build();
    }
}

package org.dynamisengine.nn.tokenizers;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.Set;

/**
 * Local GPT-2 vocab.json/merges.txt provider with the conventional end-of-text special token.
 */
public final class Gpt2TokenizerProvider implements TokenizerProvider {
    public String id() {
        return "gpt2-files";
    }

    public Tokenizer create(Map<String, String> configuration) {
        if (!configuration.keySet().equals(Set.of("vocabulary", "merges"))) {
            throw new IllegalArgumentException("gpt2-files requires vocabulary and merges paths");
        }
        try {
            return TokenizerAssets.gpt2(Path.of(configuration.get("vocabulary")), Path.of(configuration.get("merges")));
        } catch (IOException failure) {
            throw new UncheckedIOException(failure);
        }
    }
}

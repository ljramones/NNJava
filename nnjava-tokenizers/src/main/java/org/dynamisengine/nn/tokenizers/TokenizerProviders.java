package org.dynamisengine.nn.tokenizers;

import java.util.List;
import java.util.Map;
import java.util.ServiceLoader;
import java.util.TreeMap;

public final class TokenizerProviders {
    private TokenizerProviders() {
    }

    public static List<String> available() {
        return List.copyOf(providers().keySet());
    }

    public static Tokenizer create(String id, Map<String, String> configuration) {
        var provider = providers().get(id);
        if (provider == null) {
            throw new IllegalArgumentException("Unknown tokenizer provider: " + id);
        }
        return java.util.Objects.requireNonNull(provider.create(Map.copyOf(configuration)), "tokenizer");
    }

    private static Map<String, TokenizerProvider> providers() {
        var result = new TreeMap<String, TokenizerProvider>();
        for (var provider : ServiceLoader.load(TokenizerProvider.class)) {
            String id = provider.id();
            if (id == null || id.isBlank() || result.putIfAbsent(id, provider) != null) {
                throw new IllegalStateException("Invalid or duplicate tokenizer provider ID: " + id);
            }
        }
        return result;
    }
}

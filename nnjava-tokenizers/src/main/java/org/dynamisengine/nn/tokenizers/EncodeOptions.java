package org.dynamisengine.nn.tokenizers;

import java.util.Set;

/**
 * Controls are literal text unless their exact spellings are explicitly allowed.
 */
public record EncodeOptions(boolean addBos, boolean addEos, Set<String> allowedSpecial) {
    public EncodeOptions {
        allowedSpecial = Set.copyOf(allowedSpecial);
    }

    public static EncodeOptions plain() {
        return new EncodeOptions(false, false, Set.of());
    }

    public EncodeOptions withBos() {
        return new EncodeOptions(true, addEos, allowedSpecial);
    }

    public EncodeOptions withEos() {
        return new EncodeOptions(addBos, true, allowedSpecial);
    }

    public EncodeOptions allowSpecial(String... spellings) {
        return new EncodeOptions(addBos, addEos, Set.of(spellings));
    }
}

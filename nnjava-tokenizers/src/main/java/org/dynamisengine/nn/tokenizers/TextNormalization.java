package org.dynamisengine.nn.tokenizers;

import java.text.Normalizer;

/**
 * Explicit Java/JDK Unicode normalization; no implicit whitespace or punctuation cleanup.
 */
public enum TextNormalization {
    NONE, NFC, NFKC;

    static void requireScalarText(String text) {
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (Character.isHighSurrogate(c)) {
                if (++i == text.length() || !Character.isLowSurrogate(text.charAt(i))) {
                    throw new IllegalArgumentException("Unpaired UTF-16 surrogate");
                }
            } else if (Character.isLowSurrogate(c)) {
                throw new IllegalArgumentException("Unpaired UTF-16 surrogate");
            }
        }
    }

    String apply(String text) {
        return switch (this) {
            case NONE -> text;
            case NFC -> Normalizer.normalize(text, Normalizer.Form.NFC);
            case NFKC -> Normalizer.normalize(text, Normalizer.Form.NFKC);
        };
    }
}

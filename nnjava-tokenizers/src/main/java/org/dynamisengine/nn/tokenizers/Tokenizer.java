package org.dynamisengine.nn.tokenizers;

/**
 * Optional text boundary. No numerical, model, or training dependency.
 */
public interface Tokenizer extends AutoCloseable {
    Vocabulary vocabulary();

    int[] encode(String text, EncodeOptions options);

    TokenDecoder newDecoder(DecodeOptions options);

    default int[] encode(String text) {
        return encode(text, EncodeOptions.plain());
    }

    default TokenDecoder newDecoder() {
        return newDecoder(DecodeOptions.defaults());
    }

    default String decode(int[] tokenIds) {
        return decode(tokenIds, DecodeOptions.defaults());
    }

    default String decode(int[] tokenIds, DecodeOptions options) {
        var text = new StringBuilder();
        try (var decoder = newDecoder(options)) {
            for (int id : tokenIds) {
                text.append(decoder.accept(id));
            }
            return text.append(decoder.finish()).toString();
        }
    }

    @Override
    default void close() {
    }
}

package org.dynamisengine.nn.tokenizers;

/**
 * Incremental decoding. finish emits any replacement for an incomplete trailing UTF-8 sequence.
 */
public interface TokenDecoder extends AutoCloseable {
    String accept(int tokenId);

    String finish();

    @Override
    void close();
}

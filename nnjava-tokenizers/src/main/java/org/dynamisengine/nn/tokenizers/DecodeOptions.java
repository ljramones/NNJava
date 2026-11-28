package org.dynamisengine.nn.tokenizers;

public record DecodeOptions(boolean skipSpecialTokens) {
    public static DecodeOptions defaults() { return new DecodeOptions(true); }
}

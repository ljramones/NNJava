package org.dynamisengine.nn.generation;

import java.util.Arrays;

/** Immutable snapshot shared with logits/stopping hooks; prompt and continuation stay distinct. */
public final class TokenHistory {
    private final int[] tokens;
    private final int promptLength;
    public TokenHistory(int[] tokens,int promptLength) {
        this.tokens=tokens.clone(); this.promptLength=promptLength;
        if (promptLength<0 || promptLength>tokens.length || Arrays.stream(tokens).anyMatch(id -> id<0)) { throw new IllegalArgumentException("Invalid token history"); }
    }
    public int size() { return tokens.length; }
    public int promptLength() { return promptLength; }
    public int generatedCount() { return tokens.length-promptLength; }
    public int token(int index) { return tokens[index]; }
    public int[] tokens() { return tokens.clone(); }
    public int[] generatedTokens() { return Arrays.copyOfRange(tokens,promptLength,tokens.length); }
}

package org.dynamisengine.nn.tokenizers;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.regex.Pattern;

public final class PreTokenizers {
    private PreTokenizers() { }
    public static PreTokenizer wholeText() { return text -> text.isEmpty() ? List.of() : List.of(text); }
    /** Standard GPT-2-style regex using the running JDK's Unicode character classes. */
    public static PreTokenizer gpt2() {
        return regex(Pattern.compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",Pattern.UNICODE_CHARACTER_CLASS));
    }
    public static PreTokenizer regex(Pattern pattern) {
        Objects.requireNonNull(pattern);
        return text -> {
            var pieces=new ArrayList<String>(); var matcher=pattern.matcher(text); int position=0;
            while (matcher.find()) {
                if (matcher.start()!=position || matcher.end()==position) { throw new IllegalArgumentException("Pre-tokenizer regex must cover every character with nonempty matches"); }
                pieces.add(matcher.group()); position=matcher.end();
            }
            if (position!=text.length()) { throw new IllegalArgumentException("Pre-tokenizer regex left unmatched input"); }
            return List.copyOf(pieces);
        };
    }
}

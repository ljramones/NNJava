package org.dynamisengine.nn.tokenizers;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.dynamisengine.nn.formats.JsonDocuments;

/** Imports complete GPT-2 byte BPE assets; unsupported tokenizer pipelines fail explicitly. */
public final class TokenizerAssets {
    /** BOS/EOS are vocabulary spellings; special recognition still requires EncodeOptions.allowSpecial. */
    public record Options(Set<String> specialTokens, String bosToken, String eosToken) {
        public Options { specialTokens = Set.copyOf(specialTokens); }
        public static Options gpt2() { return new Options(Set.of("<|endoftext|>"), "<|endoftext|>", "<|endoftext|>"); }
        public static Options noSpecialTokens() { return new Options(Set.of(), null, null); }
    }
    private TokenizerAssets() { }

    public static BytePairTokenizer gpt2(Path vocabulary, Path merges) throws IOException {
        return gpt2(vocabulary, merges, Options.gpt2());
    }

    public static BytePairTokenizer gpt2(Path vocabulary, Path merges, Options options) throws IOException {
        var vocab = vocabulary(JsonDocuments.read(vocabulary)); var rules = new ArrayList<List<String>>();
        try (var stream = Files.newInputStream(merges)) {
            byte[] bytes = stream.readNBytes(JsonDocuments.MAX_BYTES + 1);
            if (bytes.length > JsonDocuments.MAX_BYTES) { throw new IOException("Merge file exceeds 16 MiB"); }
            String text;
            try { text = StandardCharsets.UTF_8.newDecoder().decode(java.nio.ByteBuffer.wrap(bytes)).toString(); }
            catch (java.nio.charset.CharacterCodingException failure) { throw new IOException("Invalid UTF-8 in merges", failure); }
            for (String line : text.split("\\R")) {
                if (line.isBlank() || line.startsWith("#version:")) { continue; }
                String[] pair = line.split(" ", -1);
                if (pair.length != 2 || pair[0].isEmpty() || pair[1].isEmpty()) { throw new IOException("Expected two merge symbols: " + line); }
                rules.add(List.of(pair));
            }
        }
        return build(vocab, rules, options, Set.of());
    }

    /**
     * Supports the ByteLevel BPE tokenizer.json subset: no normalization, prefix space, dropout,
     * byte fallback, non-special added tokens or template post-processing. Does not read model config.
     */
    public static BytePairTokenizer huggingFace(Path path, Options options) throws IOException {
        var document = JsonDocuments.read(path);
        if (document.get("normalizer") != null || document.get("truncation") != null || document.get("padding") != null) {
            throw new IOException("Tokenizer normalization, truncation and padding require an explicit adapter");
        }
        var pre = typed(document.get("pre_tokenizer"), "ByteLevel", "pre_tokenizer");
        if (!Boolean.FALSE.equals(pre.get("add_prefix_space")) || !Boolean.TRUE.equals(pre.get("use_regex"))) {
            throw new IOException("Only ByteLevel add_prefix_space=false/use_regex=true is supported");
        }
        typed(document.get("decoder"), "ByteLevel", "decoder");
        if (document.get("post_processor") != null) { typed(document.get("post_processor"), "ByteLevel", "post_processor"); }
        var model = typed(document.get("model"), "BPE", "model");
        if (!Set.of("type", "vocab", "merges", "dropout", "unk_token", "continuing_subword_prefix", "end_of_word_suffix", "fuse_unk", "byte_fallback", "ignore_merges").containsAll(model.keySet())) {
            throw new IOException("Unknown BPE model option");
        }
        if (model.get("dropout") != null || model.get("unk_token") != null
                || nonempty(model.get("continuing_subword_prefix")) || nonempty(model.get("end_of_word_suffix"))
                || enabledOrInvalid(model.get("fuse_unk")) || enabledOrInvalid(model.get("byte_fallback")) || enabledOrInvalid(model.get("ignore_merges"))) {
            throw new IOException("Unsupported BPE model configuration");
        }
        var vocab = vocabulary(JsonDocuments.object(model.get("vocab"), "vocab"));
        var special = new LinkedHashSet<String>();
        for (var value : JsonDocuments.array(document.getOrDefault("added_tokens", List.of()), "added_tokens")) {
            var token = JsonDocuments.object(value, "added_token");
            if (!Boolean.TRUE.equals(token.get("special")) || enabledOrInvalid(token.get("single_word"))
                    || enabledOrInvalid(token.get("lstrip")) || enabledOrInvalid(token.get("rstrip"))) {
                throw new IOException("Unsupported added-token matching rules");
            }
            String content = JsonDocuments.string(token.get("content"), "content"); int id = id(token.get("id"));
            Integer old = vocab.putIfAbsent(content, id);
            if ((old != null && old != id) || !special.add(content)) { throw new IOException("Conflicting added token"); }
        }
        var rules = new ArrayList<List<String>>();
        for (var value : JsonDocuments.array(model.get("merges"), "merges")) {
            if (value instanceof String text) {
                String[] pair = text.split(" ", -1);
                if (pair.length != 2) { throw new IOException("Invalid merge pair"); } rules.add(List.of(pair));
            } else {
                var pair = JsonDocuments.array(value, "merge");
                if (pair.size() != 2) { throw new IOException("Invalid merge pair"); }
                rules.add(List.of(JsonDocuments.string(pair.get(0), "left"), JsonDocuments.string(pair.get(1), "right")));
            }
        }
        return build(vocab, rules, options, special);
    }

    private static boolean nonempty(Object value) { return value != null && !value.equals(""); }
    private static boolean enabledOrInvalid(Object value) { return value != null && !Boolean.FALSE.equals(value); }
    private static Map<String, Object> typed(Object value, String type, String label) throws IOException {
        var fields = JsonDocuments.object(value, label);
        if (!type.equals(fields.get("type"))) { throw new IOException("Unsupported " + label + " type"); }
        if (type.equals("ByteLevel")) {
            if (!Set.of("type", "add_prefix_space", "use_regex", "trim_offsets").containsAll(fields.keySet())) { throw new IOException("Unknown ByteLevel option"); }
            for (String key : List.of("add_prefix_space", "use_regex", "trim_offsets")) {
                if (fields.containsKey(key) && !(fields.get(key) instanceof Boolean)) { throw new IOException("Expected boolean ByteLevel option: " + key); }
            }
        }
        return fields;
    }
    private static int id(Object value) throws IOException {
        long id = JsonDocuments.integer(value, "token ID");
        if (id < 0 || id > 1_000_000) { throw new IOException("Token ID exceeds supported vocabulary range"); } return (int) id;
    }
    private static Map<String, Integer> vocabulary(Map<String, Object> document) throws IOException {
        var result = new LinkedHashMap<String, Integer>();
        for (var entry : document.entrySet()) { result.put(entry.getKey(), id(entry.getValue())); } return result;
    }

    private static BytePairTokenizer build(Map<String, Integer> vocab, List<List<String>> rules, Options options, Set<String> addedSpecials) throws IOException {
        try {
            var specials = new LinkedHashSet<>(addedSpecials); specials.addAll(options.specialTokens);
            var builder = BytePairTokenizer.builder().preTokenizer(PreTokenizers.gpt2());
            var symbols = byteSymbols(); int[] byteIds = new int[256]; var used = new LinkedHashSet<String>();
            for (int b = 0; b < 256; b++) { byteIds[b] = require(vocab, symbols[b]); used.add(symbols[b]); }
            builder.byteTokenIds(byteIds);
            for (var pair : rules) {
                String left = pair.getFirst(), right = pair.getLast(), merged = left + right;
                if (!used.contains(left) || !used.contains(right) || !used.add(merged)) { throw new IOException("Noncausal or duplicate merge: " + pair); }
                builder.merge(require(vocab, left), require(vocab, right), require(vocab, merged));
            }
            for (String spelling : specials) {
                if (!used.add(spelling)) { throw new IOException("Special token overlaps normal vocabulary"); }
                builder.specialToken(require(vocab, spelling), spelling);
            }
            if (!used.equals(vocab.keySet())) { throw new IOException("Vocabulary contains tokens without a byte, merge or special-token definition"); }
            if (options.bosToken != null) { builder.bosTokenId(require(vocab, options.bosToken)); }
            if (options.eosToken != null) { builder.eosTokenId(require(vocab, options.eosToken)); }
            return builder.build();
        } catch (IllegalArgumentException | IndexOutOfBoundsException failure) { throw new IOException("Invalid byte BPE assets", failure); }
    }
    private static int require(Map<String, Integer> vocab, String spelling) throws IOException {
        Integer id = vocab.get(spelling); if (id == null) { throw new IOException("Vocabulary is missing symbol: " + spelling); } return id;
    }

    // GPT-2's reversible bytes_to_unicode alphabet. This is file-format decoding, not Unicode normalization.
    private static String[] byteSymbols() {
        String[] symbols = new String[256]; int extra = 0;
        for (int b = 0; b < 256; b++) {
            boolean direct = b >= 33 && b <= 126 || b >= 161 && b <= 172 || b >= 174;
            symbols[b] = Character.toString(direct ? b : 256 + extra++);
        }
        return symbols;
    }
}

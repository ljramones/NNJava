package org.dynamisengine.nn.tokenizers;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class TokenizerAssetsTest {
    @TempDir Path directory;
    private static final String MERGES = "#version: 0.2\nh e\nhe l\nhel l\nhell o\n";

    @Test void gpt2FilesMapTheReversibleAlphabetAndHonorRankedMerges() throws Exception {
        Path vocab = vocabulary(); Path merges = directory.resolve("merges.txt"); Files.writeString(merges, MERGES);
        var tokenizer = TokenizerAssets.gpt2(vocab, merges);
        assertArrayEquals(new int[]{259}, tokenizer.encode("hello"));
        assertEquals(220, tokenizer.encode(" ")[0]); // GPT-2's space is U+0120, vocabulary ID 220.
        for (String text : List.of("hello hello!\n", "café Ελληνικά 中文 😀", "e\u0301\0\t")) {
            assertEquals(text, tokenizer.decode(tokenizer.encode(text)));
        }
        assertArrayEquals(new int[]{260, 259, 260}, tokenizer.encode("hello", EncodeOptions.plain().withBos().withEos()));
        assertArrayEquals(new int[]{260}, tokenizer.encode("<|endoftext|>", EncodeOptions.plain().allowSpecial("<|endoftext|>")));
        assertTrue(tokenizer.encode("<|endoftext|>").length > 1);
    }

    @Test void fileProviderIsDiscoverableAndUsesTheSameAssets() throws Exception {
        Path vocab = vocabulary(); Path merges = directory.resolve("merges.txt"); Files.writeString(merges, MERGES);
        assertTrue(TokenizerProviders.available().contains("gpt2-files"));
        try (var tokenizer = TokenizerProviders.create("gpt2-files", Map.of("vocabulary", vocab.toString(), "merges", merges.toString()))) {
            assertArrayEquals(new int[]{259}, tokenizer.encode("hello"));
        }
        assertThrows(IllegalArgumentException.class, () -> TokenizerProviders.create("gpt2-files", Map.of("vocabulary", vocab.toString())));
    }

    @Test void huggingFaceStringAndArrayMergesMatchGpt2Files() throws Exception {
        String vocab = Files.readString(vocabulary());
        for (String merges : List.of("[\"h e\",\"he l\",\"hel l\",\"hell o\"]", "[[\"h\",\"e\"],[\"he\",\"l\"],[\"hel\",\"l\"],[\"hell\",\"o\"]]")) {
            Path file = directory.resolve("tokenizer.json"); Files.writeString(file, hf(vocab, merges));
            var tokenizer = TokenizerAssets.huggingFace(file, TokenizerAssets.Options.gpt2());
            assertArrayEquals(new int[]{259}, tokenizer.encode("hello")); assertEquals(261, tokenizer.vocabulary().size());
            assertEquals(" café 😀\n", tokenizer.decode(tokenizer.encode(" café 😀\n")));
        }
    }

    @Test void tokenizerJsonRejectsUnsupportedMatchingAndPipelineSemantics() throws Exception {
        String good = hf(Files.readString(vocabulary()), "[\"h e\",\"he l\",\"hel l\",\"hell o\"]");
        for (String bad : List.of(good.replace("\"normalizer\":null", "\"normalizer\":{\"type\":\"NFC\"}"),
                good.replace("\"add_prefix_space\":false", "\"add_prefix_space\":true"),
                good.replace("\"use_regex\":true", "\"use_regex\":false"), good.replace("\"dropout\":null", "\"dropout\":0.1"),
                good.replace("\"byte_fallback\":false", "\"byte_fallback\":true"),
                good.replace("\"special\":true", "\"special\":false"), good.replace("\"lstrip\":false", "\"lstrip\":true"),
                good.replace("\"post_processor\":null", "\"post_processor\":{\"type\":\"TemplateProcessing\"}"),
                good.replace("\"type\":\"ByteLevel\"", "\"type\":\"Metaspace\""))) {
            Path file = directory.resolve("bad.json"); Files.writeString(file, bad);
            assertThrows(IOException.class, () -> TokenizerAssets.huggingFace(file, TokenizerAssets.Options.gpt2()));
        }
    }

    @Test void invalidVocabularyIdsMissingBytesAndNoncausalMergesFailAtImport() throws Exception {
        Path vocab = vocabulary(); Path merges = directory.resolve("merges.txt");
        for (String text : List.of("unknown e\n", "he l\nh e\nhel l\nhell o\n", MERGES + "h e\n", "h  e\n")) {
            Files.writeString(merges, text); assertThrows(IOException.class, () -> TokenizerAssets.gpt2(vocab, merges));
        }
        Files.writeString(merges, MERGES);
        String original = Files.readString(vocab);
        Files.writeString(vocab, original.replace("\"!\":0,", "")); assertThrows(IOException.class, () -> TokenizerAssets.gpt2(vocab, merges));
        Files.writeString(vocab, original.replace("\"!\":0", "\"!\":1")); assertThrows(IOException.class, () -> TokenizerAssets.gpt2(vocab, merges));
        Files.writeString(vocab, original.replace("\"!\":0", "\"!\":1000001")); assertThrows(IOException.class, () -> TokenizerAssets.gpt2(vocab, merges));
    }

    @Test void specialTokensRequireAnExplicitDefinitionAndControlIdsMustBeSpecial() throws Exception {
        Path vocab = vocabulary(); Path merges = directory.resolve("merges.txt"); Files.writeString(merges, MERGES);
        assertThrows(IOException.class, () -> TokenizerAssets.gpt2(vocab, merges, TokenizerAssets.Options.noSpecialTokens()));
        assertThrows(IOException.class, () -> TokenizerAssets.gpt2(vocab, merges, new TokenizerAssets.Options(Set.of("<|endoftext|>"), "h", null)));
    }

    private Path vocabulary() throws IOException {
        var vocab = new LinkedHashMap<String, Integer>();
        for (int b = 33; b <= 126; b++) { vocab.put(Character.toString(b), vocab.size()); }
        for (int b = 161; b <= 172; b++) { vocab.put(Character.toString(b), vocab.size()); }
        for (int b = 174; b <= 255; b++) { vocab.put(Character.toString(b), vocab.size()); }
        for (int i = 0; vocab.size() < 256; i++) { vocab.put(Character.toString(256 + i), vocab.size()); }
        for (String token : List.of("he", "hel", "hell", "hello", "<|endoftext|>")) { vocab.put(token, vocab.size()); }
        var json = new StringBuilder("{");
        vocab.forEach((key, id) -> { if (json.length() > 1) { json.append(','); } json.append('"').append(key.replace("\\", "\\\\").replace("\"", "\\\"")).append("\":").append(id); });
        Path path = directory.resolve("vocab.json"); Files.writeString(path, json.append('}')); return path;
    }
    private static String hf(String vocab, String merges) {
        return "{\"normalizer\":null,\"pre_tokenizer\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"use_regex\":true},"
                + "\"decoder\":{\"type\":\"ByteLevel\"},\"post_processor\":null,"
                + "\"added_tokens\":[{\"id\":260,\"content\":\"<|endoftext|>\",\"special\":true,\"lstrip\":false,\"rstrip\":false,\"single_word\":false}],"
                + "\"model\":{\"type\":\"BPE\",\"dropout\":null,\"byte_fallback\":false,\"vocab\":" + vocab + ",\"merges\":" + merges + "}}";
    }
}

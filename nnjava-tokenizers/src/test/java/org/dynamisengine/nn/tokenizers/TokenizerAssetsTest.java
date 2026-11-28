package org.dynamisengine.nn.tokenizers;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class TokenizerAssetsTest {
    private static final String MERGES = "#version: 0.2\nh e\nhe l\nhel l\nhell o\n";
    @TempDir
    Path directory;

    private static String hf(String vocab, String merges) {
        return "{\"normalizer\":null,\"pre_tokenizer\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"use_regex\":true},"
                + "\"decoder\":{\"type\":\"ByteLevel\"},\"post_processor\":null,"
                + "\"added_tokens\":[{\"id\":260,\"content\":\"<|endoftext|>\",\"special\":true,\"lstrip\":false,\"rstrip\":false,\"single_word\":false}],"
                + "\"model\":{\"type\":\"BPE\",\"dropout\":null,\"byte_fallback\":false,\"vocab\":" + vocab + ",\"merges\":" + merges + "}}";
    }

    @Test
    void importsDeclaredDigitsByteLevelSequenceAndRejectsOtherSequences() throws Exception {
        String pre = "\"pre_tokenizer\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"use_regex\":true}";
        String sequence = "\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":[{\"type\":\"Digits\",\"individual_digits\":true},{\"type\":\"ByteLevel\",\"add_prefix_space\":false,\"use_regex\":true}]}";
        String source = hf(Files.readString(vocabulary()), "[\"h e\",\"he l\",\"hel l\",\"hell o\"]").replace(pre, sequence);
        Path file = directory.resolve("sequence.json");
        Files.writeString(file, source);
        var tokenizer = TokenizerAssets.huggingFace(file, TokenizerAssets.Options.gpt2());
        assertArrayEquals(new int[]{259}, tokenizer.encode("hello"));
        assertEquals(4, tokenizer.encode("1234").length);
        Files.writeString(file, source.replace("\"individual_digits\":true", "\"individual_digits\":false"));
        assertThrows(IOException.class, () -> TokenizerAssets.huggingFace(file, TokenizerAssets.Options.gpt2()));
    }

    @Test
    void tokenizerJsonCanDeclareASparseAlphabetButUnavailableBytesFailExplicitly() throws Exception {
        String source = hf("{\"a\":0,\"b\":1,\"ab\":2,\"<eos>\":3}", "[\"a b\"]").replace("\"id\":260", "\"id\":3").replace("<|endoftext|>", "<eos>");
        Path file = directory.resolve("sparse.json");
        Files.writeString(file, source);
        var tokenizer = TokenizerAssets.huggingFace(file, new TokenizerAssets.Options(Set.of(), null, "<eos>"));
        assertArrayEquals(new int[]{2}, tokenizer.encode("ab"));
        assertEquals(4, tokenizer.vocabulary().size());
        assertThrows(IllegalArgumentException.class, () -> tokenizer.encode("c"));
    }

    @Test
    void gpt2FilesMapTheReversibleAlphabetAndHonorRankedMerges() throws Exception {
        Path vocab = vocabulary();
        Path merges = directory.resolve("merges.txt");
        Files.writeString(merges, MERGES);
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

    @Test
    void fileProviderIsDiscoverableAndUsesTheSameAssets() throws Exception {
        Path vocab = vocabulary();
        Path merges = directory.resolve("merges.txt");
        Files.writeString(merges, MERGES);
        assertTrue(TokenizerProviders.available().contains("gpt2-files"));
        try (var tokenizer = TokenizerProviders.create("gpt2-files", Map.of("vocabulary", vocab.toString(), "merges", merges.toString()))) {
            assertArrayEquals(new int[]{259}, tokenizer.encode("hello"));
        }
        assertThrows(IllegalArgumentException.class, () -> TokenizerProviders.create("gpt2-files", Map.of("vocabulary", vocab.toString())));
    }

    @Test
    void huggingFaceStringAndArrayMergesMatchGpt2Files() throws Exception {
        String vocab = Files.readString(vocabulary());
        for (String merges : List.of("[\"h e\",\"he l\",\"hel l\",\"hell o\"]", "[[\"h\",\"e\"],[\"he\",\"l\"],[\"hel\",\"l\"],[\"hell\",\"o\"]]")) {
            Path file = directory.resolve("tokenizer.json");
            Files.writeString(file, hf(vocab, merges));
            var tokenizer = TokenizerAssets.huggingFace(file, TokenizerAssets.Options.gpt2());
            assertArrayEquals(new int[]{259}, tokenizer.encode("hello"));
            assertEquals(261, tokenizer.vocabulary().size());
            assertEquals(" café 😀\n", tokenizer.decode(tokenizer.encode(" café 😀\n")));
        }
    }

    @Test
    void tokenizerJsonRejectsUnsupportedMatchingAndPipelineSemantics() throws Exception {
        String good = hf(Files.readString(vocabulary()), "[\"h e\",\"he l\",\"hel l\",\"hell o\"]");
        for (String bad : List.of(good.replace("\"normalizer\":null", "\"normalizer\":{\"type\":\"NFC\"}"),
                good.replace("\"add_prefix_space\":false", "\"add_prefix_space\":true"),
                good.replace("\"use_regex\":true", "\"use_regex\":false"), good.replace("\"dropout\":null", "\"dropout\":0.1"),
                good.replace("\"byte_fallback\":false", "\"byte_fallback\":true"),
                good.replace("\"special\":true", "\"special\":false"), good.replace("\"lstrip\":false", "\"lstrip\":true"),
                good.replace("\"post_processor\":null", "\"post_processor\":{\"type\":\"TemplateProcessing\"}"),
                good.replace("\"type\":\"ByteLevel\"", "\"type\":\"Metaspace\""))) {
            Path file = directory.resolve("bad.json");
            Files.writeString(file, bad);
            assertThrows(IOException.class, () -> TokenizerAssets.huggingFace(file, TokenizerAssets.Options.gpt2()));
        }
    }

    @Test
    void invalidVocabularyIdsMissingBytesAndNoncausalMergesFailAtImport() throws Exception {
        Path vocab = vocabulary();
        Path merges = directory.resolve("merges.txt");
        for (String text : List.of("unknown e\n", "he l\nh e\nhel l\nhell o\n", MERGES + "h e\n", "h  e\n")) {
            Files.writeString(merges, text);
            assertThrows(IOException.class, () -> TokenizerAssets.gpt2(vocab, merges));
        }
        Files.writeString(merges, MERGES);
        String original = Files.readString(vocab);
        Files.writeString(vocab, original.replace("\"!\":0,", ""));
        assertThrows(IOException.class, () -> TokenizerAssets.gpt2(vocab, merges));
        Files.writeString(vocab, original.replace("\"!\":0", "\"!\":1"));
        assertThrows(IOException.class, () -> TokenizerAssets.gpt2(vocab, merges));
        Files.writeString(vocab, original.replace("\"!\":0", "\"!\":1000001"));
        assertThrows(IOException.class, () -> TokenizerAssets.gpt2(vocab, merges));
    }

    @Test
    void specialTokensRequireAnExplicitDefinitionAndControlIdsMustBeSpecial() throws Exception {
        Path vocab = vocabulary();
        Path merges = directory.resolve("merges.txt");
        Files.writeString(merges, MERGES);
        assertThrows(IOException.class, () -> TokenizerAssets.gpt2(vocab, merges, TokenizerAssets.Options.noSpecialTokens()));
        assertThrows(IOException.class, () -> TokenizerAssets.gpt2(vocab, merges, new TokenizerAssets.Options(Set.of("<|endoftext|>"), "h", null)));
    }

    private Path vocabulary() throws IOException {
        var vocab = new LinkedHashMap<String, Integer>();
        for (int b = 33; b <= 126; b++) {
            vocab.put(Character.toString(b), vocab.size());
        }
        for (int b = 161; b <= 172; b++) {
            vocab.put(Character.toString(b), vocab.size());
        }
        for (int b = 174; b <= 255; b++) {
            vocab.put(Character.toString(b), vocab.size());
        }
        for (int i = 0; vocab.size() < 256; i++) {
            vocab.put(Character.toString(256 + i), vocab.size());
        }
        for (String token : List.of("he", "hel", "hell", "hello", "<|endoftext|>")) {
            vocab.put(token, vocab.size());
        }
        var json = new StringBuilder("{");
        vocab.forEach((key, id) -> {
            if (json.length() > 1) {
                json.append(',');
            }
            json.append('"').append(key.replace("\\", "\\\\").replace("\"", "\\\"")).append("\":").append(id);
        });
        Path path = directory.resolve("vocab.json");
        Files.writeString(path, json.append('}'));
        return path;
    }
}

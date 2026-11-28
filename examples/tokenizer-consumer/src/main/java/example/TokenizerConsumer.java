package example;

import java.util.Map;
import org.dynamisengine.nn.tokenizers.BytePairTokenizer;
import org.dynamisengine.nn.tokenizers.EncodeOptions;
import org.dynamisengine.nn.tokenizers.TokenizerProviders;
import org.dynamisengine.nn.tokenizers.TokenizerAssets;
import java.nio.file.Files;

/** This consumer runs without model or numerical modules; format parsing uses Jackson Core. */
public final class TokenizerConsumer {
    public static void main(String[] args) throws Exception {
        try (var tokenizer=BytePairTokenizer.builder().merge('h','e',256).merge(256,'l',257).merge(257,'l',258).merge(258,'o',259)
                .specialToken(260,"<bos>").specialToken(261,"<eos>").bosTokenId(260).eosTokenId(261).build();
             var decoder=tokenizer.newDecoder()) {
            String text="hello😀"; int[] ids=tokenizer.encode(text,EncodeOptions.plain().withBos().withEos()); var output=new StringBuilder();
            for (int id:ids) { output.append(decoder.accept(id)); } output.append(decoder.finish());
            if (!text.equals(output.toString()) || !text.equals(tokenizer.decode(ids))) { throw new AssertionError("Packaged BPE/Unicode round trip failed"); }
        }
        try (var tokenizer=TokenizerProviders.create("utf8-bytes",Map.of("normalization","NFC"))) {
            if (!"é".equals(tokenizer.decode(tokenizer.encode("e\u0301")))) { throw new AssertionError("Tokenizer provider discovery failed"); }
        }
        verifyFiles();
        System.out.println("Tokenizer-only JPMS consumer passed without model or numerical modules");
    }

    private static void verifyFiles() throws Exception {
        var directory = Files.createTempDirectory("nnjava-tokenizer-"); var vocabulary = directory.resolve("vocab.json"); var merges = directory.resolve("merges.txt");
        var json = new StringBuilder("{"); int extra = 0;
        for (int b = 0; b < 256; b++) {
            boolean direct = b >= 33 && b <= 126 || b >= 161 && b <= 172 || b >= 174;
            String symbol = Character.toString(direct ? b : 256 + extra++);
            if (b > 0) { json.append(','); } json.append('"').append(symbol.replace("\\", "\\\\").replace("\"", "\\\"")).append("\":").append(b);
        }
        json.append(",\"he\":256,\"<|endoftext|>\":257}");
        try {
            Files.writeString(vocabulary, json); Files.writeString(merges, "#version: 0.2\nh e\n");
            try (var tokenizer = TokenizerAssets.gpt2(vocabulary, merges)) {
                if (!java.util.Arrays.equals(new int[]{256}, tokenizer.encode("he")) || !"hé 😀".equals(tokenizer.decode(tokenizer.encode("hé 😀")))) {
                    throw new AssertionError("Packaged GPT-2 assets failed");
                }
            }
            try (var provider = TokenizerProviders.create("gpt2-files", Map.of("vocabulary", vocabulary.toString(), "merges", merges.toString()))) {
                if (provider.vocabulary().eosTokenId().orElseThrow() != 257) { throw new AssertionError("Packaged file provider failed"); }
            }
        } finally { Files.deleteIfExists(vocabulary); Files.deleteIfExists(merges); Files.deleteIfExists(directory); }
    }
}

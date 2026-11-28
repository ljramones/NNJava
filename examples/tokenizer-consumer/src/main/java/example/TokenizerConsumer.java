package example;

import java.util.Map;
import org.dynamisengine.nn.tokenizers.BytePairTokenizer;
import org.dynamisengine.nn.tokenizers.EncodeOptions;
import org.dynamisengine.nn.tokenizers.TokenizerProviders;

/** This consumer runs with only the tokenizer jar and java.base on its dependency path. */
public final class TokenizerConsumer {
    public static void main(String[] args) {
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
        System.out.println("Tokenizer-only JPMS consumer passed without model or numerical modules");
    }
}

package org.dynamisengine.nn.tokenizers;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class BytePairTokenizerTest {
    @Test void completeByteVocabularyRoundTripsUnicodeWhitespaceAndControlText() {
        var tokenizer=BytePairTokenizer.builder().build();
        for (String text:List.of("","hello\n\t world  ","café Ελληνικά 中文 😀","e\u0301","\0\r\n<|eos|>")) {
            assertEquals(text,tokenizer.decode(tokenizer.encode(text)));
            assertArrayEquals(text.getBytes(StandardCharsets.UTF_8),bytes(tokenizer,tokenizer.encode(text)));
        }
        assertEquals(256,tokenizer.vocabulary().size()); assertTrue(tokenizer.vocabulary().bosTokenId().isEmpty());
    }

    @Test void rankedMergesProduceExpectedTokensAndBytes() {
        var tokenizer=hello().build();
        assertArrayEquals(new int[]{259},tokenizer.encode("hello")); assertEquals("hello",tokenizer.decode(new int[]{259}));
        assertArrayEquals("hello".getBytes(StandardCharsets.UTF_8),tokenizer.vocabulary().tokenBytes(259));
        assertArrayEquals(new int[]{259,32,259},tokenizer.encode("hello hello"));
    }

    @Test void vocabularyLookupCannotBypassTheRankedMergePath() {
        var tokenizer=BytePairTokenizer.builder().merge('b','c',256).merge('a','b',257).merge(257,'c',258).build();
        assertArrayEquals("abc".getBytes(StandardCharsets.UTF_8),tokenizer.vocabulary().tokenBytes(258));
        assertArrayEquals(new int[]{'a',256},tokenizer.encode("abc"));
    }

    @Test void equalRankOccurrencesMergeLeftmostAndStalePairsAreDiscarded() {
        var leftmost=BytePairTokenizer.builder().merge('a','a',256).merge(256,'a',257).build();
        assertArrayEquals(new int[]{257},leftmost.encode("aaa"));
        var tokenizer=BytePairTokenizer.builder().merge('a','b',256).merge('b','c',257).merge(256,'c',258).merge('c','d',259).merge(258,'d',260).build();
        assertArrayEquals(new int[]{260},tokenizer.encode("abcd"));
    }

    @Test void optimizedMergingMatchesAnIndependentRescanOracle() {
        int[][] merges={{'a','b',256},{'b','c',257},{256,'c',258},{'c','d',259},{258,'d',260},{'a','a',261},{261,'b',262}};
        var builder=BytePairTokenizer.builder(); for (var merge:merges) { builder.merge(merge[0],merge[1],merge[2]); }
        var tokenizer=builder.build(); var random=new Random(42);
        for (int trial=0;trial<300;trial++) {
            var text=new StringBuilder(); for (int i=0,n=random.nextInt(40);i<n;i++) { text.append((char)('a'+random.nextInt(4))); }
            String value=text.toString(); assertArrayEquals(oracle(value,merges),tokenizer.encode(value)); assertEquals(value,tokenizer.decode(tokenizer.encode(value)));
        }
    }

    @Test void specialParsingIsOptInAndBosEosAreNeverImplicit() {
        var tokenizer=hello().specialToken(260,"<s>").specialToken(261,"</s>").bosTokenId(260).eosTokenId(261).build();
        assertArrayEquals(new int[]{259},tokenizer.encode("hello"));
        assertArrayEquals(new int[]{260,259,261},tokenizer.encode("hello",EncodeOptions.plain().withBos().withEos()));
        assertFalse(Arrays.stream(tokenizer.encode("<s>hello</s>")).anyMatch(id -> id==260 || id==261));
        assertArrayEquals(new int[]{260,259,261},tokenizer.encode("<s>hello</s>",EncodeOptions.plain().allowSpecial("<s>","</s>")));
        assertEquals("hello",tokenizer.decode(new int[]{260,259,261}));
        assertEquals("<s>hello</s>",tokenizer.decode(new int[]{260,259,261},new DecodeOptions(false)));
        assertArrayEquals(new int[]{260,260},tokenizer.encode("<s>",EncodeOptions.plain().withBos().allowSpecial("<s>")));
        assertThrows(IllegalArgumentException.class, () -> tokenizer.encode("hello",EncodeOptions.plain().allowSpecial("<unknown>")));
    }

    @Test void specialTokenMatchingQuotesMetacharactersAndUsesLongestAllowedSpelling() {
        var tokenizer=BytePairTokenizer.builder().specialToken(256,"[x]").specialToken(257,"[x]tail").build();
        assertArrayEquals(new int[]{257},tokenizer.encode("[x]tail",EncodeOptions.plain().allowSpecial("[x]","[x]tail")));
        assertArrayEquals(new int[]{256,'t','a','i','l'},tokenizer.encode("[x]tail",EncodeOptions.plain().allowSpecial("[x]")));
        assertEquals("[x]tail",tokenizer.decode(tokenizer.encode("[x]tail")));
    }

    @Test void incrementalDecoderHoldsUtf8UntilACompleteCodePointArrives() {
        var tokenizer=BytePairTokenizer.builder().specialToken(256,"<s>").build();
        try (var decoder=tokenizer.newDecoder()) {
            assertEquals("",decoder.accept(0xf0)); assertEquals("",decoder.accept(0x9f)); assertEquals("",decoder.accept(256));
            assertEquals("",decoder.accept(0x98)); assertEquals("😀",decoder.accept(0x80)); assertEquals("",decoder.finish());
            assertEquals("",decoder.finish()); assertThrows(IllegalStateException.class, () -> decoder.accept('a'));
        }
        try (var decoder=tokenizer.newDecoder()) { assertEquals("",decoder.accept(0xe2)); assertEquals("�",decoder.finish()); }
    }

    @Test void decoderHandlesMultibyteMergedTokensAndMalformedSequencesLikeJavaUtf8() {
        var tokenizer=BytePairTokenizer.builder().merge(0xf0,0x9f,256).merge(256,0x98,257).build();
        try (var decoder=tokenizer.newDecoder()) { assertEquals("",decoder.accept(257)); assertEquals("😀",decoder.accept(0x80)); assertEquals("",decoder.finish()); }
        var plain=BytePairTokenizer.builder().build(); var random=new Random(91);
        for (int trial=0;trial<100;trial++) {
            byte[] raw=new byte[random.nextInt(80)]; random.nextBytes(raw); int[] ids=new int[raw.length];
            for (int i=0;i<raw.length;i++) { ids[i]=raw[i]&255; }
            assertEquals(new String(raw,StandardCharsets.UTF_8),plain.decode(ids));
        }
    }

    @Test void normalizationIsExplicitAndDoesNotChangeSpecialTokenRecognition() {
        var plain=BytePairTokenizer.builder().build(); var nfc=BytePairTokenizer.builder().normalization(TextNormalization.NFC).build();
        var nfkc=BytePairTokenizer.builder().normalization(TextNormalization.NFKC).build();
        assertEquals("e\u0301",plain.decode(plain.encode("e\u0301"))); assertEquals("é",nfc.decode(nfc.encode("e\u0301")));
        assertEquals("A",nfkc.decode(nfkc.encode("Ａ")));
        assertThrows(IllegalArgumentException.class, () -> plain.encode("\ud800")); assertThrows(IllegalArgumentException.class, () -> plain.encode("\udc00"));
    }

    @Test void pretokenizersPreserveCoverageAndPreventCrossPieceMerges() {
        var whole=BytePairTokenizer.builder().merge('a',' ',256).merge(256,'b',257).build();
        var split=BytePairTokenizer.builder().merge('a',' ',256).merge(256,'b',257).preTokenizer(PreTokenizers.gpt2()).build();
        assertArrayEquals(new int[]{257},whole.encode("a b")); assertArrayEquals(new int[]{'a',' ','b'},split.encode("a b"));
        String value="We're  here!\n中文 １２３ 😀"; assertEquals(value,split.decode(split.encode(value)));
        var bad=BytePairTokenizer.builder().preTokenizer(PreTokenizers.regex(Pattern.compile("[a-z]+"))).build();
        assertThrows(IllegalArgumentException.class, () -> bad.encode("a 1"));
        assertThrows(IllegalArgumentException.class, () -> BytePairTokenizer.builder().preTokenizer(text -> List.of("a")).build().encode("abc"));
        assertThrows(IllegalArgumentException.class, () -> BytePairTokenizer.builder().preTokenizer(text -> List.of("")).build().encode("a"));
    }

    @Test void buildersAndVocabularyDefensivelyCopyMutableInputs() {
        int[] bytes=java.util.stream.IntStream.range(0,256).map(i -> 255-i).toArray();
        var builder=BytePairTokenizer.builder().byteTokenIds(bytes); var first=builder.build(); bytes[0]=0;
        builder.specialToken(256,"<eos>").eosTokenId(256); var second=builder.build();
        assertArrayEquals(new int[]{255-'a'},first.encode("a")); assertEquals(256,first.vocabulary().size()); assertEquals(257,second.vocabulary().size());
        byte[] copy=first.vocabulary().tokenBytes(255-'a'); copy[0]=0; assertEquals("a",first.decode(first.encode("a")));
        assertThrows(UnsupportedOperationException.class, () -> second.vocabulary().specialTokens().put("new",4));
    }

    @Test void invalidVocabularyMergesAndControlMetadataAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> BytePairTokenizer.builder().byteTokenIds(1,2));
        int[] duplicates=new int[256]; assertThrows(IllegalArgumentException.class, () -> BytePairTokenizer.builder().byteTokenIds(duplicates).build());
        assertThrows(RuntimeException.class, () -> BytePairTokenizer.builder().merge(256,'a',257).build());
        assertThrows(IllegalArgumentException.class, () -> BytePairTokenizer.builder().merge('a','b',256).merge('a','b',257).build());
        assertThrows(IllegalArgumentException.class, () -> BytePairTokenizer.builder().specialToken(256,"").build());
        assertThrows(IllegalArgumentException.class, () -> BytePairTokenizer.builder().specialToken(256,"x").specialToken(257,"x").build());
        assertThrows(IllegalArgumentException.class, () -> BytePairTokenizer.builder().bosTokenId(0).build());
        assertThrows(IllegalArgumentException.class, () -> BytePairTokenizer.builder().build().encode("x",EncodeOptions.plain().withEos()));
    }

    @Test void unknownTokenIdsAndClosedDecodersFailWithoutCorruptingAnotherDecoder() {
        var tokenizer=BytePairTokenizer.builder().build(); var decoder=tokenizer.newDecoder();
        assertThrows(IndexOutOfBoundsException.class, () -> decoder.accept(256)); assertEquals("a",decoder.accept('a'));
        decoder.close(); decoder.close(); assertThrows(IllegalStateException.class,decoder::finish);
        assertEquals("ok",tokenizer.decode(tokenizer.encode("ok")));
    }

    @Test void tokenizerProvidersWorkWithoutAnyNumericalOrModelModule() {
        assertTrue(TokenizerProviders.available().contains("utf8-bytes"));
        try (var tokenizer=TokenizerProviders.create("utf8-bytes",Map.of("normalization","NFC"))) { assertEquals("é",tokenizer.decode(tokenizer.encode("e\u0301"))); }
        assertThrows(IllegalArgumentException.class, () -> TokenizerProviders.create("unknown",Map.of()));
        assertThrows(IllegalArgumentException.class, () -> TokenizerProviders.create("utf8-bytes",Map.of("unsupported","true")));
    }

    @Test void immutableTokenizersCanBeSharedButDecodersAreThreadConfined() throws Exception {
        var tokenizer=hello().build();
        try (var decoder=tokenizer.newDecoder(); var executor=java.util.concurrent.Executors.newFixedThreadPool(2)) {
            assertEquals("hello😀",executor.submit(() -> tokenizer.decode(tokenizer.encode("hello😀"))).get());
            executor.submit(() -> assertThrows(IllegalStateException.class, () -> decoder.accept('a'))).get();
        }
    }

    private static BytePairTokenizer.Builder hello() { return BytePairTokenizer.builder().merge('h','e',256).merge(256,'l',257).merge(257,'l',258).merge(258,'o',259); }
    private static byte[] bytes(Tokenizer tokenizer,int[] ids) {
        var bytes=new java.io.ByteArrayOutputStream(); for (int id:ids) { bytes.writeBytes(tokenizer.vocabulary().tokenBytes(id)); } return bytes.toByteArray();
    }
    private static int[] oracle(String text,int[][] merges) {
        var ids=new ArrayList<Integer>(); for (byte b:text.getBytes(StandardCharsets.UTF_8)) { ids.add(b&255); }
        while (true) {
            int rank=Integer.MAX_VALUE,index=-1;
            for (int i=0;i<ids.size()-1;i++) for (int r=0;r<merges.length;r++) {
                if (ids.get(i)==merges[r][0] && ids.get(i+1)==merges[r][1] && r<rank) { rank=r; index=i; }
            }
            if (index<0) { return ids.stream().mapToInt(Integer::intValue).toArray(); }
            ids.set(index,merges[rank][2]); ids.remove(index+1);
        }
    }
}

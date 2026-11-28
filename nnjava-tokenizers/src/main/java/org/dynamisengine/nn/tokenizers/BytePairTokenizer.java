package org.dynamisengine.nn.tokenizers;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.regex.Pattern;

/** Ranked byte-level BPE with explicit special-token policy; defaults to a complete byte alphabet. */
public final class BytePairTokenizer implements Tokenizer {
    private record Rule(int left,int right,int result) { }
    private record Merge(int rank,int left,int right,int leftId,int rightId) implements Comparable<Merge> {
        public int compareTo(Merge other) { int rankOrder=Integer.compare(rank,other.rank); return rankOrder==0 ? Integer.compare(left,other.left) : rankOrder; }
    }
    private final Vocabulary vocabulary;
    private final int[] byteIds;
    private final List<Rule> rules;
    private final Map<Long,Integer> ranks;
    private final TextNormalization normalization;
    private final PreTokenizer preTokenizer;

    private BytePairTokenizer(Builder builder) {
        byteIds=builder.byteIds.clone(); rules=List.copyOf(builder.rules);
        normalization=Objects.requireNonNull(builder.normalization); preTokenizer=Objects.requireNonNull(builder.preTokenizer);
        int alphabetSize=0;
        for (int id:byteIds) { if (id < -1) { throw new IllegalArgumentException("Byte IDs must be nonnegative or -1 for an absent byte"); } if (id>=0) { alphabetSize++; } }
        int size=Math.addExact(alphabetSize,Math.addExact(rules.size(),builder.specials.size()));
        byte[][] pieces=new byte[size][]; var normalPieces=new HashMap<String,Integer>(); var ranked=new HashMap<Long,Integer>();
        for (int b=0;b<256;b++) { if (byteIds[b]>=0) { define(pieces,byteIds[b],new byte[]{(byte)b},normalPieces); } }
        for (int rank=0;rank<rules.size();rank++) {
            var rule=rules.get(rank); int left=Objects.checkIndex(rule.left,size),right=Objects.checkIndex(rule.right,size);
            if (pieces[left]==null || pieces[right]==null) { throw new IllegalArgumentException("Merge components must be bytes or earlier merged tokens"); }
            byte[] merged=Arrays.copyOf(pieces[left],Math.addExact(pieces[left].length,pieces[right].length));
            System.arraycopy(pieces[right],0,merged,pieces[left].length,pieces[right].length);
            define(pieces,rule.result,merged,normalPieces);
            if (ranked.put(pair(left,right),rank)!=null) { throw new IllegalArgumentException("Duplicate BPE merge pair"); }
        }
        builder.specials.forEach((id,text) -> {
            Objects.checkIndex(id,size); TextNormalization.requireScalarText(text);
            if (text.isEmpty() || pieces[id]!=null) { throw new IllegalArgumentException("Invalid or duplicate special token ID"); }
            pieces[id]=text.getBytes(StandardCharsets.UTF_8);
        });
        ranks=Map.copyOf(ranked); vocabulary=new Vocabulary(pieces,builder.specials,builder.bos,builder.eos);
    }

    private static void define(byte[][] pieces,int id,byte[] bytes,Map<String,Integer> seen) {
        Objects.checkIndex(id,pieces.length);
        if (pieces[id]!=null || seen.put(new String(bytes,StandardCharsets.ISO_8859_1),id)!=null) { throw new IllegalArgumentException("Duplicate normal token ID or byte spelling"); }
        pieces[id]=bytes;
    }
    public static Builder builder() { return new Builder(); }
    @Override public Vocabulary vocabulary() { return vocabulary; }
    @Override public TokenDecoder newDecoder(DecodeOptions options) { return new Utf8TokenDecoder(vocabulary,options); }

    @Override public int[] encode(String text,EncodeOptions options) {
        Objects.requireNonNull(text); Objects.requireNonNull(options); TextNormalization.requireScalarText(text);
        if (!vocabulary.specialTokens().keySet().containsAll(options.allowedSpecial())) { throw new IllegalArgumentException("Unknown allowed special token spelling"); }
        var result=new ArrayList<Integer>();
        int bos=options.addBos() ? vocabulary.bosTokenId().orElseThrow(() -> new IllegalArgumentException("No BOS token configured")) : -1;
        int eos=options.addEos() ? vocabulary.eosTokenId().orElseThrow(() -> new IllegalArgumentException("No EOS token configured")) : -1;
        if (bos!=-1) { result.add(bos); }
        int start=0;
        if (!options.allowedSpecial().isEmpty()) {
            String pattern=String.join("|",options.allowedSpecial().stream().sorted(Comparator.comparingInt(String::length).reversed().thenComparing(Comparator.naturalOrder())).map(Pattern::quote).toList());
            var matcher=Pattern.compile(pattern).matcher(text);
            while (matcher.find()) {
                ordinary(text.substring(start,matcher.start()),result); result.add(vocabulary.specialTokens().get(matcher.group())); start=matcher.end();
            }
        }
        ordinary(text.substring(start),result);
        if (eos!=-1) { result.add(eos); }
        return result.stream().mapToInt(Integer::intValue).toArray();
    }

    private void ordinary(String text,ArrayList<Integer> result) {
        String normalized=normalization.apply(text); int offset=0;
        for (String piece:Objects.requireNonNull(preTokenizer.split(normalized))) {
            if (piece==null || piece.isEmpty() || !normalized.startsWith(piece,offset)) { throw new IllegalArgumentException("Pre-tokenizer pieces do not cover the normalized input"); }
            TextNormalization.requireScalarText(piece); encodePiece(piece.getBytes(StandardCharsets.UTF_8),result); offset=Math.addExact(offset,piece.length());
        }
        if (offset!=normalized.length()) { throw new IllegalArgumentException("Pre-tokenizer omitted input characters"); }
    }

    private void encodePiece(byte[] bytes,ArrayList<Integer> output) {
        if (bytes.length==0) { return; }
        int[] ids=new int[bytes.length],previous=new int[bytes.length],next=new int[bytes.length]; var queue=new PriorityQueue<Merge>();
        for (int i=0;i<bytes.length;i++) {
            ids[i]=byteIds[bytes[i]&255];
            if (ids[i]<0) { throw new IllegalArgumentException("Tokenizer vocabulary has no token for byte 0x"+Integer.toHexString(bytes[i]&255)); }
            previous[i]=i-1; next[i]=i+1==bytes.length ? -1 : i+1;
        }
        for (int i=0;i<bytes.length-1;i++) { offer(i,i+1,ids,queue); }
        while (!queue.isEmpty()) {
            var merge=queue.remove(); int left=merge.left,right=merge.right;
            if (next[left]!=right || ids[left]!=merge.leftId || ids[right]!=merge.rightId) { continue; }
            ids[left]=rules.get(merge.rank).result; ids[right]=-1; next[left]=next[right];
            if (next[left]>=0) { previous[next[left]]=left; }
            offer(previous[left],left,ids,queue); offer(left,next[left],ids,queue);
        }
        for (int i=0;i!=-1;i=next[i]) { output.add(ids[i]); }
    }
    private void offer(int left,int right,int[] ids,PriorityQueue<Merge> queue) {
        if (left<0 || right<0) { return; }
        Integer rank=ranks.get(pair(ids[left],ids[right]));
        if (rank!=null) { queue.add(new Merge(rank,left,right,ids[left],ids[right])); }
    }
    private static long pair(int left,int right) { return (long)left<<32 | Integer.toUnsignedLong(right); }

    public static final class Builder {
        private int[] byteIds=java.util.stream.IntStream.range(0,256).toArray();
        private final List<Rule> rules=new ArrayList<>();
        private final Map<Integer,String> specials=new LinkedHashMap<>();
        private int bos=-1,eos=-1;
        private TextNormalization normalization=TextNormalization.NONE;
        private PreTokenizer preTokenizer=PreTokenizers.wholeText();
        private Builder() { }
        /** A 256-entry byte-to-ID table; -1 explicitly marks an unavailable byte, which fails encoding. */
        public Builder byteTokenIds(int... ids) { if (ids.length!=256) { throw new IllegalArgumentException("Exactly 256 byte IDs required"); } byteIds=ids.clone(); return this; }
        public Builder merge(int left,int right,int result) { rules.add(new Rule(left,right,result)); return this; }
        public Builder specialToken(int id,String spelling) {
            if (specials.putIfAbsent(id,Objects.requireNonNull(spelling))!=null) { throw new IllegalArgumentException("Duplicate special token ID"); } return this;
        }
        public Builder bosTokenId(int value) { bos=value; return this; }
        public Builder eosTokenId(int value) { eos=value; return this; }
        public Builder normalization(TextNormalization value) { normalization=Objects.requireNonNull(value); return this; }
        public Builder preTokenizer(PreTokenizer value) { preTokenizer=Objects.requireNonNull(value); return this; }
        public BytePairTokenizer build() { return new BytePairTokenizer(this); }
    }
}

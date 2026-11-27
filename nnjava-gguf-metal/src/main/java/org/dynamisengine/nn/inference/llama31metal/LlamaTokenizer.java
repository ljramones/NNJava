package org.dynamisengine.nn.inference.llama31metal;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.regex.Pattern;

import static org.dynamisengine.nn.inference.llama31metal.Llama3CharacterClasses.*;

/** GGUF byte-level BPE for the Llama 3.1 gpt2/llama-bpe vocabulary.
 * Instances own their vocabulary and can be reused after the checkpoint file is closed. */
public final class LlamaTokenizer {
    private static final int NORMAL = 1, CONTROL = 3;
    private static final char[] BYTE_CHARS = byteAlphabet();
    private static final int[] CHAR_BYTES = inverseAlphabet();
    private final String[] tokens;
    private final int[] types;
    private final int[] byteIds = new int[256], mergedIds;
    private final int bos, eos, endOfText, endOfMessage, endOfTurn;
    private final Map<String,Integer> tokenIds;
    private final Map<Long,Integer> mergeRanks;
    private final Pattern controls;

    public static LlamaTokenizer load(Path gguf) throws IOException {
        try (var file = Gguf.open(gguf)) { return from(file); }
    }

    static LlamaTokenizer from(Gguf file) {
        if (!"gpt2".equals(file.text("tokenizer.ggml.model")) || !"llama-bpe".equals(file.text("tokenizer.ggml.pre"))) {
            throw new IllegalArgumentException("Expected a GGUF gpt2/llama-bpe tokenizer");
        }
        if (!file.flag("tokenizer.ggml.add_bos_token",true) || !file.flag("tokenizer.ggml.clean_spaces",true)
                || file.flag("tokenizer.ggml.add_eos_token",false) || file.flag("tokenizer.ggml.add_sep_token",false)
                || file.flag("tokenizer.ggml.add_space_prefix",false) || file.flag("tokenizer.ggml.remove_extra_whitespaces",false)
                || file.flag("tokenizer.ggml.normalizer.nfc",false) || file.flag("tokenizer.ggml.normalizer.lowercase",false)
                || file.flag("tokenizer.ggml.normalizer.strip_accents",false)) {
            throw new IllegalArgumentException("Unsupported tokenizer normalization or special-token flags");
        }
        return new LlamaTokenizer(file.strings("tokenizer.ggml.tokens"), file.integers("tokenizer.ggml.token_type"),
                file.strings("tokenizer.ggml.merges"), file.integer("tokenizer.ggml.bos_token_id"),
                file.integer("tokenizer.ggml.eos_token_id"));
    }

    private LlamaTokenizer(String[] vocabulary, int[] tokenTypes, String[] merges, int bos, int eos) {
        tokens=vocabulary.clone(); types=tokenTypes.clone();
        if (tokens.length!=128256 || types.length!=tokens.length) throw new IllegalArgumentException("Expected Llama 3.1 vocabulary");
        tokenIds=HashMap.newHashMap(tokens.length);
        var special=new ArrayList<String>();
        for (int i=0; i<tokens.length; i++) {
            String value=tokens[i];
            if (value.isEmpty() || tokenIds.put(value,i)!=null) throw new IllegalArgumentException("Invalid vocabulary piece "+i);
            if (types[i]==CONTROL) special.add(value);
            else if (types[i]!=NORMAL) throw new IllegalArgumentException("Unsupported token type "+types[i]);
        }
        this.bos=requiredControl("<|begin_of_text|>"); this.eos=eos;
        endOfText=requiredControl("<|end_of_text|>"); endOfMessage=requiredControl("<|eom_id|>"); endOfTurn=requiredControl("<|eot_id|>");
        if (this.bos!=bos || eos!=endOfTurn) throw new IllegalArgumentException("Unexpected Llama 3.1 BOS/EOS metadata");
        for (int b=0; b<256; b++) byteIds[b]=requiredNormal(String.valueOf(BYTE_CHARS[b]));
        mergeRanks=HashMap.newHashMap(merges.length);
        mergedIds=new int[merges.length];
        for (int rank=0; rank<merges.length; rank++) {
            String merge=merges[rank]; int space=merge.indexOf(' ');
            if (space<=0 || space!=merge.lastIndexOf(' ') || space==merge.length()-1) throw new IllegalArgumentException("Malformed merge "+rank);
            String a=merge.substring(0,space), b=merge.substring(space+1);
            long key=pair(requiredNormal(a),requiredNormal(b));
            if (mergeRanks.put(key,rank)!=null) throw new IllegalArgumentException("Duplicate merge "+rank);
            mergedIds[rank]=requiredNormal(a+b);
        }
        special.sort(Comparator.comparingInt(String::length).reversed());
        controls=Pattern.compile(String.join("|",special.stream().map(Pattern::quote).toList()));
    }

    private int requiredControl(String text) {
        Integer id=tokenIds.get(text);
        if (id==null || types[id]!=CONTROL) throw new IllegalArgumentException("Missing control token "+text);
        return id;
    }
    private int requiredNormal(String text) {
        Integer id=tokenIds.get(text);
        if (id==null || types[id]!=NORMAL) throw new IllegalArgumentException("Missing byte-level vocabulary component: "+text);
        return id;
    }

    public int bosTokenId() { return bos; }
    public int eosTokenId() { return eos; }
    public int vocabularySize() { return tokens.length; }
    public boolean isEndOfGeneration(int token) {
        Objects.checkIndex(token,tokens.length);
        return token==endOfText || token==endOfMessage || token==endOfTurn;
    }

    /** Encode Unicode scalar text. A requested BOS is prepended even if the text starts with a literal BOS. */
    public int[] encode(String text, boolean addBos, boolean parseSpecial) {
        Objects.requireNonNull(text);
        var output=new ArrayList<Integer>();
        if (addBos) output.add(bos);
        int start=0;
        if (parseSpecial) {
            var matcher=controls.matcher(text);
            while (matcher.find()) {
                ordinary(text,start,matcher.start(),output);
                output.add(tokenIds.get(matcher.group()));
                start=matcher.end();
            }
        }
        ordinary(text,start,text.length(),output);
        return output.stream().mapToInt(Integer::intValue).toArray();
    }

    /** Raw bytes for a vocabulary item. A token may contain just part of a UTF-8 code point. */
    public byte[] tokenBytes(int token, boolean renderSpecial) {
        Objects.checkIndex(token,tokens.length);
        String piece=tokens[token];
        if (types[token]==CONTROL) return renderSpecial ? piece.getBytes(StandardCharsets.UTF_8) : new byte[0];
        byte[] bytes=new byte[piece.length()];
        for (int j=0; j<bytes.length; j++) {
            char c=piece.charAt(j);
            if (c>=CHAR_BYTES.length || CHAR_BYTES[c]<0) throw new IllegalArgumentException("Malformed byte-level token "+token);
            bytes[j]=(byte)CHAR_BYTES[c];
        }
        return bytes;
    }

    /** Decode with the checkpoint's native punctuation/contraction cleanup. Incomplete UTF-8 is replaced with U+FFFD. Use tokenBytes for verbatim output. */
    public String decode(int[] ids, boolean renderSpecial) {
        var bytes=new ByteArrayOutputStream();
        for (int id:ids) bytes.writeBytes(tokenBytes(id,renderSpecial));
        return bytes.toString(StandardCharsets.UTF_8)
                .replace(" .",".").replace(" ?","?").replace(" !","!").replace(" ,",",")
                .replace(" ' ","'").replace(" 'm","'m").replace(" 's","'s").replace(" 've","'ve").replace(" 're","'re");
    }

    private void ordinary(String text, int first, int last, ArrayList<Integer> output) {
        if (first==last) return;
        int[] cps=text.substring(first,last).codePoints().toArray();
        int[] kinds=new int[cps.length];
        for (int i=0; i<cps.length; i++) {
            if (cps[i]>=0xd800 && cps[i]<=0xdfff) throw new IllegalArgumentException("Unpaired UTF-16 surrogate");
            kinds[i]=Llama3CharacterClasses.of(cps[i]);
        }
        int offset=first;
        for (int i=0; i<cps.length;) {
            int end=pieceEnd(cps,kinds,i);
            int next=offset;
            for (int j=i; j<end; j++) next+=Character.charCount(cps[j]);
            encodePiece(text.substring(offset,next).getBytes(StandardCharsets.UTF_8),output);
            i=end;offset=next;
        }
    }

    private static int pieceEnd(int[] cp, int[] kind, int i) {
        int n=cp.length;
        if (cp[i]=='\'' && i+1<n) {
            int c=cp[i+1]|32;
            if (c=='s'||c=='t'||c=='m'||c=='d') return i+2;
            if (i+2<n) {
                int next=cp[i+2]|32;
                if ((c=='r'&&next=='e')||(c=='v'&&next=='e')||(c=='l'&&next=='l')) return i+3;
            }
        }
        int end=i;
        if (kind[i]==LETTER || (cp[i]!='\r' && cp[i]!='\n' && kind[i]!=NUMBER && i+1<n && kind[i+1]==LETTER)) {
            end=i+1;
            while (end<n && kind[end]==LETTER) end++;
            return end;
        }
        if (kind[i]==NUMBER) {
            end=i+1;
            while (end<n && end-i<3 && kind[end]==NUMBER) end++;
            return end;
        }
        if (kind[i]==OTHER || (cp[i]==' ' && i+1<n && kind[i+1]==OTHER)) {
            end=i+1;
            while (end<n && kind[end]==OTHER) end++;
            while (end<n && (cp[end]=='\r' || cp[end]=='\n')) end++;
            return end;
        }
        int afterNewline=-1;
        while (end<n && kind[end]==SPACE) {
            if (cp[end]=='\n'||cp[end]=='\r') afterNewline=end+1;
            end++;
        }
        if (afterNewline>=0) return afterNewline;
        return end<n && end>i+1 ? end-1 : end;
    }

    private void encodePiece(byte[] bytes, ArrayList<Integer> output) {
        char[] glyphs=new char[bytes.length];
        for (int i=0; i<bytes.length; i++) glyphs[i]=BYTE_CHARS[bytes[i]&255];
        Integer exact=tokenIds.get(new String(glyphs));
        if (exact!=null && types[exact]==NORMAL) { output.add(exact); return; }

        int n=bytes.length;
        int[] ids=new int[n], previous=new int[n], next=new int[n];
        var pairs=new PriorityQueue<Merge>();
        for (int i=0; i<n; i++) {
            ids[i]=byteIds[bytes[i]&255]; previous[i]=i-1; next[i]=i==n-1 ? -1 : i+1;
        }
        for (int i=0; i<n-1; i++) offer(i,i+1,ids,pairs);
        while (!pairs.isEmpty()) {
            Merge m=pairs.remove();
            int l=m.left(),r=m.right();
            if (next[l]!=r || ids[l]!=m.leftId() || ids[r]!=m.rightId()) continue;
            ids[l]=mergedIds[m.rank()]; ids[r]=-1; next[l]=next[r];
            if (next[l]>=0) previous[next[l]]=l;
            offer(previous[l],l,ids,pairs);offer(l,next[l],ids,pairs);
        }
        for (int i=0; i>=0; i=next[i]) output.add(ids[i]);
    }

    private record Merge(int rank, int left, int right, int leftId, int rightId) implements Comparable<Merge> {
        public int compareTo(Merge other) {
            int result=Integer.compare(rank,other.rank);
            return result==0 ? Integer.compare(left,other.left) : result;
        }
    }

    private void offer(int left, int right, int[] ids, PriorityQueue<Merge> queue) {
        if (left<0 || right<0) return;
        Integer rank=mergeRanks.get(pair(ids[left],ids[right]));
        if (rank!=null) queue.add(new Merge(rank,left,right,ids[left],ids[right]));
    }
    private static long pair(int left, int right) { return (long)left << 32 | Integer.toUnsignedLong(right); }

    private static char[] byteAlphabet() {
        char[] result=new char[256];
        int extra=256;
        for (int b=0; b<256; b++) {
            boolean printed=(b>=33 && b<=126) || (b>=161 && b<=172) || b>=174;
            result[b]=(char)(printed ? b : extra++);
        }
        return result;
    }
    private static int[] inverseAlphabet() {
        int[] result=new int[512]; Arrays.fill(result,-1);
        for (int i=0; i<256; i++) result[BYTE_CHARS[i]]=i;
        return result;
    }
}

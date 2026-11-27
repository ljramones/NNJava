package org.dynamisengine.nn.inference.llama31metal;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.List;

/** Check a real checkpoint against the frozen, independently generated CPU oracle. */
public final class TokenizerParity {
    private static void check(boolean ok, String cause) {
        if (!ok) throw new AssertionError(cause);
    }

    public static void main(String[] args) throws Exception {
        if (args.length!=2) throw new IllegalArgumentException("CHECKPOINT ORACLE_DIRECTORY");
        var vocab=LlamaTokenizer.load(Path.of(args[0]));
        var source=Path.of(args[1]);
        var hex=HexFormat.of();
        var rows=Files.readAllLines(source.resolve("oracle.tsv"));
        for (String row:rows.subList(1,rows.size())) {
            String[] f=row.split("\t",-1);
            if (f.length!=7) throw new AssertionError("Malformed oracle row");
            String text=new String(hex.parseHex(f[3]),StandardCharsets.UTF_8);
            int[] actual=vocab.encode(text,"1".equals(f[1]),"1".equals(f[2]));
            int[] expected=f[4].isEmpty()?new int[0]:Arrays.stream(f[4].split(",")).mapToInt(Integer::parseInt).toArray();
            check(Arrays.equals(actual,expected),f[0]+"/"+f[1]+"/"+f[2]+" expected "+Arrays.toString(expected)+" actual "+Arrays.toString(actual));
            for (int render=0;render<=1;render++) {
                String decoded=vocab.decode(actual,render!=0);
                check(f[5+render].equals(hex.formatHex(decoded.getBytes(StandardCharsets.UTF_8))),"detokenize "+f[0]);
            }
        }
        List<String> digests=Files.readAllLines(source.resolve("pieces.sha256"));
        List<String> eog=new ArrayList<>();
        for (int render=0;render<=1;render++) {
            var hash=MessageDigest.getInstance("SHA-256");
            for (int token=0;token<vocab.vocabularySize();token++) {
                byte[] bytes=vocab.tokenBytes(token,render!=0);
                for (int shift:new int[]{0,8,16,24}) hash.update((byte)(bytes.length>>>shift));
                hash.update(bytes);
                if (render==0 && vocab.isEndOfGeneration(token)) eog.add(Integer.toString(token));
            }
            check(digests.get(render).equals(hex.formatHex(hash.digest())),"token piece digest "+render);
        }
        check(eog.equals(Files.readAllLines(source.resolve("eog.txt"))),"EOG coverage");
        try { vocab.encode("invalid\ud800",false,false); throw new AssertionError("Unexpected surrogate acceptance"); }
        catch (IllegalArgumentException expected) { }
        System.out.printf("%d encode and %d decode comparisons, %d token pieces in both modes, EOG %s%n",
                rows.size()-1,2*(rows.size()-1),vocab.vocabularySize(),eog);
    }
}

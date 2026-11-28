package org.dynamisengine.nn.tokenizers;

import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.Objects;

/** Buffers incomplete UTF-8 bytes rather than emitting a replacement at token boundaries. */
final class Utf8TokenDecoder implements TokenDecoder {
    private final Thread owner=Thread.currentThread();
    private final Vocabulary vocabulary;
    private final DecodeOptions options;
    private final java.nio.charset.CharsetDecoder decoder=StandardCharsets.UTF_8.newDecoder()
            .onMalformedInput(CodingErrorAction.REPLACE).onUnmappableCharacter(CodingErrorAction.REPLACE);
    private byte[] pending=new byte[0];
    private boolean finished,closed;
    Utf8TokenDecoder(Vocabulary vocabulary,DecodeOptions options) { this.vocabulary=vocabulary; this.options=Objects.requireNonNull(options); }
    @Override public String accept(int tokenId) {
        ensureOpen();
        if (finished) { throw new IllegalStateException("Decoder is finished"); }
        boolean special=vocabulary.isSpecial(tokenId);
        if (special && options.skipSpecialTokens()) { return ""; }
        return decode(vocabulary.tokenBytes(tokenId),false);
    }
    @Override public String finish() {
        ensureOpen();
        if (finished) { return ""; }
        String text=decode(new byte[0],true); finished=true; return text;
    }
    private String decode(byte[] bytes,boolean end) {
        var input=ByteBuffer.allocate(Math.addExact(pending.length,bytes.length)); input.put(pending).put(bytes).flip();
        var output=CharBuffer.allocate(Math.addExact(input.remaining(),4));
        var result=decoder.decode(input,output,end);
        if (!result.isUnderflow()) { throw new IllegalStateException("Unexpected UTF-8 decoding result: "+result); }
        pending=new byte[input.remaining()]; input.get(pending);
        if (end && !decoder.flush(output).isUnderflow()) { throw new IllegalStateException("Unexpected UTF-8 flush result"); }
        return output.flip().toString();
    }
    private void ensureOpen() {
        if (Thread.currentThread()!=owner) { throw new IllegalStateException("Token decoder is thread-confined"); }
        if (closed) { throw new IllegalStateException("Token decoder is closed"); }
    }
    @Override public void close() {
        if (Thread.currentThread()!=owner) { throw new IllegalStateException("Token decoder is thread-confined"); }
        closed=true; pending=new byte[0];
    }
}

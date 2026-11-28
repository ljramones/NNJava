package org.dynamisengine.nn.tokenizers;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalInt;

/**
 * Immutable ID/byte vocabulary with explicit special-token metadata.
 */
public final class Vocabulary {
    private final byte[][] pieces;
    private final Map<String, Integer> specials;
    private final boolean[] special;
    private final int bos, eos;

    Vocabulary(byte[][] values, Map<Integer, String> controls, int bos, int eos) {
        pieces = new byte[values.length][];
        special = new boolean[values.length];
        for (int i = 0; i < values.length; i++) {
            pieces[i] = Objects.requireNonNull(values[i], "Unassigned vocabulary ID " + i).clone();
        }
        var names = new LinkedHashMap<String, Integer>();
        controls.forEach((id, text) -> {
            Objects.checkIndex(id, pieces.length);
            special[id] = true;
            if (names.put(text, id) != null) {
                throw new IllegalArgumentException("Duplicate special token spelling");
            }
        });
        specials = Map.copyOf(names);
        this.bos = bos;
        this.eos = eos;
        for (int id : new int[]{bos, eos}) {
            if (id != -1 && (id < 0 || id >= pieces.length || !special[id])) {
                throw new IllegalArgumentException("BOS/EOS must name configured special tokens");
            }
        }
    }

    public int size() {
        return pieces.length;
    }

    public byte[] tokenBytes(int id) {
        return pieces[Objects.checkIndex(id, pieces.length)].clone();
    }

    public boolean isSpecial(int id) {
        return special[Objects.checkIndex(id, pieces.length)];
    }

    public Map<String, Integer> specialTokens() {
        return specials;
    }

    public OptionalInt bosTokenId() {
        return bos == -1 ? OptionalInt.empty() : OptionalInt.of(bos);
    }

    public OptionalInt eosTokenId() {
        return eos == -1 ? OptionalInt.empty() : OptionalInt.of(eos);
    }
}

package org.dynamisengine.nn.generation;

/** Adapter boundary for a borrowed model and owned single-sequence inference caches. */
public interface AutoregressiveModel {
    int vocabularySize();
    int contextLength();
    Cursor openSession();

    /** Returns owned host FP32 logits for the last appended token; no training modules required. */
    interface Cursor extends AutoCloseable {
        int length();
        float[] prefill(int[] tokens);
        float[] decode(int tokenId);
        @Override void close();
    }
}

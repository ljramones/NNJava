package org.dynamisengine.nn.generation;

import java.util.Objects;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;
import org.dynamisengine.nn.transformer.CausalLanguageModel;

/** Host-logit and tensor-logit adapters over the public causal-model cache API. */
public final class CausalModels {
    private CausalModels() { }
    /** Tensor-only adapter: no logit downloads or token uploads between decode steps. */
    public static TensorAutoregressiveModel adaptDevice(CausalLanguageModel model) {
        Objects.requireNonNull(model);
        return new TensorAutoregressiveModel() {
            public org.dynamisengine.nn.runtime.ExecutionContext executionContext() { return model.executionContext(); }
            public int vocabularySize() { return model.definition().vocabularySize(); }
            public int contextLength() { return model.definition().contextLength(); }
            public Cursor openSession() {
                var session = model.newSession(1);
                return new Cursor() {
                    public int length() { return Math.toIntExact(session.length()); }
                    public Tensor prefill(Tensor tokens) { try (var logits = session.prefill(tokens)) { return last(logits); } }
                    public Tensor decode(Tensor token) { try (var logits = session.decode(token)) { return last(logits); } }
                    private Tensor last(Tensor logits) {
                        long time = logits.shape().dimension(1);
                        try (var row = Ops.slice(logits, 1, time - 1, time)) { return Ops.reshape(row, Shape.of(model.definition().vocabularySize())); }
                    }
                    public void close() { session.close(); }
                };
            }
        };
    }
    public static AutoregressiveModel adapt(CausalLanguageModel model) {
        Objects.requireNonNull(model);
        return new AutoregressiveModel() {
            public int vocabularySize() { return model.definition().vocabularySize(); }
            public int contextLength() { return model.definition().contextLength(); }
            public Cursor openSession() {
                var context=model.executionContext(); var session=model.newSession(1);
                return new Cursor() {
                    public int length() { return Math.toIntExact(session.length()); }
                    public float[] prefill(int[] tokens) {
                        for (int token:tokens) { check(token); }
                        try (var input=context.tensor(tokens,Shape.of(1,tokens.length)); var logits=session.prefill(input)) { return last(logits); }
                    }
                    public float[] decode(int tokenId) {
                        check(tokenId);
                        try (var input=context.tensor(new int[]{tokenId},Shape.of(1,1)); var logits=session.decode(input)) { return last(logits); }
                    }
                    private void check(int id) { if (id<0 || id>=model.definition().vocabularySize()) { throw new IllegalArgumentException("Token outside model vocabulary"); } }
                    public void close() { session.close(); }
                    private float[] last(Tensor logits) {
                        long length=logits.shape().dimension(1);
                        try (var row=Ops.slice(logits,1,length-1,length)) { return row.toFloatArray(); }
                    }
                };
            }
        };
    }
}

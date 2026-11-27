package org.dynamisengine.nn.examples;

import org.dynamisengine.nn.nn.*;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;
import org.dynamisengine.nn.training.AdamW;
import org.dynamisengine.nn.training.Losses;
import org.dynamisengine.nn.training.Metrics;
import org.dynamisengine.nn.transformer.GatedFeedForward;
import org.dynamisengine.nn.transformer.MultiHeadAttention;

/**
 * A separate Maven consumer assembles its own architecture from the library JAR.
 */
public final class TransformerTrainingExample {
    public static void main(String[] args) {
        try (var context = ExecutionContext.cpu(); var model = new ConsumerDecoder(context); var optimizer = new AdamW(model.namedParameters(), .025f);
             var tokens = context.tensor(new int[]{0, 1, 2, 1, 2, 3}, Shape.of(2, 3)); var labels = context.tensor(new int[]{1, 2, 3, 2, 3, 4}, Shape.of(2, 3))) {
            float first = 0, last = 0;
            for (int i = 0; i < 60; i++) {
                optimizer.zeroGrad();
                try (var tape = context.gradientTape(); var logits = model.forward(tokens); var loss = Losses.crossEntropy(logits, labels)) {
                    last = loss.item();
                    if (i == 0) {
                        first = last;
                    }
                    tape.backward(loss);
                }
                optimizer.step();
            }
            model.training(false);
            try (var prediction = model.forward(tokens)) {
                double accuracy = Metrics.accuracy(prediction, labels, -100);
                if (last >= first * .15f || accuracy < .9) {
                    throw new AssertionError("Consumer training failed");
                }
                System.out.printf("Consumer transformer CE: %.6f -> %.6f; token accuracy: %.3f%n", first, last, accuracy);
            }
        }
    }

    static final class ConsumerDecoder extends AbstractModule implements UnaryModule {
        private final Embedding embedding;
        private final RmsNorm norm1, norm2;
        private final MultiHeadAttention attention;
        private final GatedFeedForward feedForward;
        private final Linear output;

        ConsumerDecoder(ExecutionContext c) {
            super(c);
            embedding = register("embedding", new Embedding(c, 5, 4, 1));
            norm1 = register("norm1", new RmsNorm(c, 4, 1e-5f));
            attention = register("attention", new MultiHeadAttention(c, 4, 2, 1, 2));
            norm2 = register("norm2", new RmsNorm(c, 4, 1e-5f));
            feedForward = register("feedForward", new GatedFeedForward(c, 4, 8, 3));
            output = register("output", new Linear(c, embedding.weight(), false));
        }

        @Override
        public Tensor forward(Tensor tokens) {
            try (var x = embedding.forward(tokens); var n1 = norm1.forward(x); var attended = attention.forward(n1); var residual = Ops.add(x, attended);
                 var n2 = norm2.forward(residual); var feed = feedForward.forward(n2); var hidden = Ops.add(residual, feed)) {
                return output.forward(hidden);
            }
        }
    }
}

package example;

import java.nio.file.Files;
import java.util.Map;
import org.dynamisengine.nn.data.DataLoader;
import org.dynamisengine.nn.data.TensorDataset;
import org.dynamisengine.nn.nn.*;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.*;
import org.dynamisengine.nn.training.*;
import org.dynamisengine.nn.transformer.*;

/**
 * A separate Maven consumer assembles its own architecture from the library JAR.
 */
public final class LibraryConsumer {
    public static void main(String[] args) throws Exception {
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
        verifyTrainer();
    }

    private static void verifyTrainer() throws Exception {
        var definition = NeuralNetwork.builder().inputFeatures(2).seed(42)
                .dense("hidden",8,Activation.Kind.TANH).dense("output",2).define();
        var optimizer = Optimizers.adamW().learningRate(.03f).define();
        var path = Files.createTempFile("nnjava-consumer-training-", ".nnj");
        try {
            try (var context = ExecutionContext.cpu(); var model = definition.build(context); var restored = definition.build(context);
                 var input = context.tensor(new float[]{0,0,0,1,1,0,1,1},Shape.of(4,2));
                 var target = context.tensor(new int[]{0,1,1,0},Shape.of(4));
                 var data = new TensorDataset(Map.of("input",input,"target",target));
                 var loader = DataLoader.builder(context,data).batchSize(4).build();
                 var resumedLoader = DataLoader.builder(context,data).batchSize(4).build();
                 var trainer = Trainer.builder().model(model).optimizer(optimizer).build();
                 var resumed = Trainer.builder().model(restored).optimizer(optimizer).build()) {
                double first = trainer.evaluate(loader).loss(); trainer.fit(loader,200);
                trainer.save(path,loader); resumed.restore(path,resumedLoader);
                var result = resumed.evaluate(resumedLoader);
                if (result.loss() >= first*.1 || result.score()!=1 || !trainer.progress().equals(resumed.progress())) {
                    throw new AssertionError("Consumer Trainer or checkpoint failed");
                }
                System.out.printf("Consumer Trainer CE: %.6f -> %.6f; restored updates: %d%n",first,result.loss(),resumed.progress().steps());
            }
        } finally { Files.deleteIfExists(path); }
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

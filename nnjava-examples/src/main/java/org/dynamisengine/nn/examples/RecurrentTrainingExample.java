package org.dynamisengine.nn.examples;

import org.dynamisengine.nn.data.DataLoader;
import org.dynamisengine.nn.data.TensorDataset;
import org.dynamisengine.nn.io.ModelDefinitions;
import org.dynamisengine.nn.nn.Layers;
import org.dynamisengine.nn.nn.NeuralNetwork;
import org.dynamisengine.nn.nn.RecurrentCell;
import org.dynamisengine.nn.nn.SequenceOutput;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.training.Optimizers;
import org.dynamisengine.nn.training.Trainer;

import java.nio.file.Files;
import java.util.Arrays;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Comparable RNN/LSTM/GRU delayed-recall tasks through the public fluent API.
 */
public final class RecurrentTrainingExample {
    private RecurrentTrainingExample() {
    }

    public static void main(String[] args) throws Exception {
        run(args.length == 0 ? RecurrentCell.Kind.LSTM : RecurrentCell.Kind.valueOf(args[0].toUpperCase(java.util.Locale.ROOT)));
    }

    public static void run(RecurrentCell.Kind kind) throws Exception {
        int time = kind == RecurrentCell.Kind.RNN ? 4 : 8;
        var memory = switch (kind) {
            case RNN -> Layers.rnn(8);
            case LSTM -> Layers.lstm(8);
            case GRU -> Layers.gru(8);
        };
        var definition = NeuralNetwork.builder().name(kind.name().toLowerCase(java.util.Locale.ROOT) + "-recall").inputShape(time, 1).seed(42)
                .layer("memory", memory.output(SequenceOutput.LAST)).dense("logits", 2).define();
        var optimizer = Optimizers.adamW().learningRate(.02f).define();
        var directory = Files.createTempDirectory("nnjava-recurrent-");
        var description = directory.resolve("model.nnjd");
        var checkpoint = directory.resolve("training.nnj");
        try {
            ModelDefinitions.write(description, definition.specification());
            var restoredDefinition = NeuralNetwork.Definition.fromSpec(ModelDefinitions.read(description));
            try (var context = ExecutionContext.cpu(); var model = definition.build(context); var restored = restoredDefinition.build(context);
                 var training = recall(context, 24, time, 17); var heldOut = recall(context, 12, time, 97);
                 var loader = DataLoader.builder(context, training).batchSize(6).shuffle(true).seed(24).build();
                 var resumedLoader = DataLoader.builder(context, training).batchSize(6).shuffle(true).seed(24).build();
                 var validation = DataLoader.builder(context, heldOut).batchSize(5).build();
                 var trainer = Trainer.builder().model(model).optimizer(optimizer).clipGlobalNorm(1).build();
                 var resumed = Trainer.builder().model(restored).optimizer(optimizer).clipGlobalNorm(1).build()) {
                double first = trainer.evaluate(validation).loss();
                trainer.fit(loader, 20);
                trainer.save(checkpoint, loader);
                resumed.restore(checkpoint, resumedLoader);
                trainer.fit(loader, 60);
                resumed.fit(resumedLoader, 60);
                var result = resumed.evaluate(validation);
                try (var batch = heldOut.batch(context, new int[]{0, 1, 2, 3}); var input = batch.get("input");
                     var expected = model.forward(input); var actual = restored.forward(input)) {
                    if (!Arrays.equals(expected.toFloatArray(), actual.toFloatArray()) || result.loss() >= first * .15 || result.score() < .9) {
                        throw new AssertionError(kind + " learning, validation, or checkpoint resume failed");
                    }
                }
                System.out.printf("Fluent %s recall (%d steps) CE: %.6f -> %.6f; held-out accuracy: %.3f; %d updates; exact checkpoint resume%n",
                        kind, time, first, result.loss(), result.score(), resumed.progress().steps());
            }
        } finally {
            Files.deleteIfExists(description);
            Files.deleteIfExists(checkpoint);
            Files.deleteIfExists(directory);
        }
    }

    private static TensorDataset recall(ExecutionContext context, int count, int time, long seed) {
        float[] values = new float[count * time];
        int[] labels = new int[count];
        var random = new SplittableRandom(seed);
        for (int row = 0; row < count; row++) {
            int label = row % 2;
            labels[row] = label;
            values[row * time] = (label == 0 ? -1f : 1f) * (float) (.8 + random.nextDouble() * .4);
            for (int step = 1; step < time; step++) {
                values[row * time + step] = (float) (random.nextDouble() - .5) * .3f;
            }
        }
        try (var input = context.tensor(values, Shape.of(count, time, 1)); var target = context.tensor(labels, Shape.of(count))) {
            return new TensorDataset(Map.of("input", input, "target", target));
        }
    }
}

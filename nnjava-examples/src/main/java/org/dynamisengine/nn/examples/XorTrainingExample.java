package org.dynamisengine.nn.examples;

import org.dynamisengine.nn.io.ModelDefinitions;
import org.dynamisengine.nn.io.StateArchive;
import org.dynamisengine.nn.nn.Activation;
import org.dynamisengine.nn.nn.NeuralNetwork;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.training.AdamW;
import org.dynamisengine.nn.training.Losses;
import org.dynamisengine.nn.training.Metrics;

import java.nio.file.Files;

/**
 * Build, train, and restore a small MLP entirely through the public fluent API.
 */
public final class XorTrainingExample {
    private XorTrainingExample() {
    }

    public static void main(String[] args) throws Exception {
        var definition = NeuralNetwork.builder().name("xor").inputFeatures(2).seed(42)
                .dense("hidden", 8, Activation.Kind.TANH).dense("logits", 2).define();
        var directory = Files.createTempDirectory("nnjava-xor-");
        var description = directory.resolve("model.nnjd");
        var weights = directory.resolve("model.nnj");
        try {
            ModelDefinitions.write(description, definition.specification());
            try (var context = ExecutionContext.cpu(); var model = definition.build(context);
                 var optimizer = new AdamW(model.namedParameters(), .03f);
                 var input = context.tensor(new float[]{0, 0, 0, 1, 1, 0, 1, 1}, Shape.of(4, 2));
                 var labels = context.tensor(new int[]{0, 1, 1, 0}, Shape.of(4))) {
                float first = 0, last = 0;
                for (int step = 0; step < 200; step++) {
                    optimizer.zeroGrad();
                    try (var tape = context.gradientTape(); var logits = model.forward(input); var loss = Losses.crossEntropy(logits, labels)) {
                        last = loss.item();
                        if (step == 0) {
                            first = last;
                        }
                        tape.backward(loss);
                    }
                    optimizer.step();
                }
                model.training(false);
                StateArchive.saveWeights(weights, model);
                var restoredDefinition = NeuralNetwork.Definition.fromSpec(ModelDefinitions.read(description));
                try (var restored = restoredDefinition.build(context)) {
                    StateArchive.read(weights).restoreModel(restored);
                    try (var logits = restored.forward(input)) {
                        double accuracy = Metrics.accuracy(logits, labels, -100);
                        if (last >= first * .1f || accuracy != 1) {
                            throw new AssertionError("XOR training failed");
                        }
                        System.out.printf("Fluent MLP loss: %.6f -> %.6f; accuracy: %.3f; definition and weights restored%n", first, last, accuracy);
                    }
                }
            }
        } finally {
            Files.deleteIfExists(description);
            Files.deleteIfExists(weights);
            Files.deleteIfExists(directory);
        }
    }
}

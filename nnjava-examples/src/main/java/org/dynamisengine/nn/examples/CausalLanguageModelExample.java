package org.dynamisengine.nn.examples;

import java.nio.file.Files;
import java.util.Arrays;
import org.dynamisengine.nn.io.ModelDefinitions;
import org.dynamisengine.nn.io.StateArchive;
import org.dynamisengine.nn.nn.NormalizationKind;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.training.AdamW;
import org.dynamisengine.nn.training.Losses;
import org.dynamisengine.nn.training.Metrics;
import org.dynamisengine.nn.transformer.CausalLanguageModel;

/** Tiny caller-configured decoder: train token IDs, restore its definition/weights, and use cached inference. */
public final class CausalLanguageModelExample {
    private CausalLanguageModelExample() { }

    public static void main(String[] args) throws Exception {
        var definition = CausalLanguageModel.builder().name("example-decoder").vocabularySize(5)
                .modelWidth(4).blockCount(2).contextLength(8).attention(a -> a.queryHeads(2).keyValueHeads(1).rotaryPositions())
                .feedForward(f -> f.hiddenWidth(8).swiGlu()).normalization(NormalizationKind.RMS)
                .tieInputOutputEmbeddings(true).seed(42).define();
        var directory = Files.createTempDirectory("nnjava-decoder-");
        var description = directory.resolve("model.nnjd"); var weights = directory.resolve("model.nnj");
        try {
            ModelDefinitions.write(description, definition.specification());
            try (var context = ExecutionContext.cpu(); var model = definition.build(context);
                    var optimizer = new AdamW(model.namedParameters(), .025f);
                    var tokens = context.tensor(new int[] {0, 1, 2, 1, 2, 3}, Shape.of(2, 3));
                    var labels = context.tensor(new int[] {1, 2, 3, 2, 3, 4}, Shape.of(2, 3))) {
                float first = 0, last = 0;
                for (int step = 0; step < 100; step++) {
                    optimizer.zeroGrad();
                    try (var tape = context.gradientTape(); var logits = model.forward(tokens); var loss = Losses.crossEntropy(logits, labels)) {
                        last = loss.item(); if (step == 0) { first = last; } tape.backward(loss);
                    }
                    optimizer.step();
                }
                model.training(false); StateArchive.saveWeights(weights, model);
                var restoredDefinition = CausalLanguageModel.Definition.fromSpec(ModelDefinitions.read(description));
                try (var restored = restoredDefinition.build(context)) {
                    StateArchive.read(weights).restoreModel(restored);
                    try (var full = restored.forward(tokens); var session = restored.newSession(2);
                            var prefix = Ops.slice(tokens, 1, 0, 2); var suffix = Ops.slice(tokens, 1, 2, 3);
                            var ignored = session.prefill(prefix); var cached = session.decode(suffix);
                            var expected = Ops.slice(full, 1, 2, 3); var next = Ops.argmax(cached, -1, false)) {
                        float[] a = expected.toFloatArray(), b = cached.toFloatArray();
                        for (int i = 0; i < a.length; i++) {
                            if (Math.abs(a[i] - b[i]) > 2e-5f) { throw new AssertionError("Cached logits differ"); }
                        }
                        double accuracy = Metrics.accuracy(full, labels, -100);
                        if (last >= first * .1f || accuracy < .9) { throw new AssertionError("Decoder training failed"); }
                        System.out.printf("Fluent decoder CE: %.6f -> %.6f; accuracy: %.3f; cache length: %d; next token IDs: %s%n",
                                first, last, accuracy, session.length(), Arrays.toString(next.toIntArray()));
                    }
                }
            }
        } finally {
            Files.deleteIfExists(description); Files.deleteIfExists(weights); Files.deleteIfExists(directory);
        }
    }
}

package example;

import java.nio.file.Files;
import java.util.Arrays;
import org.dynamisengine.nn.io.ModelDefinitions;
import org.dynamisengine.nn.io.StateArchive;
import org.dynamisengine.nn.nn.NeuralNetwork;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.transformer.CausalLanguageModel;

/** Fluent model definitions, archives and decoder sessions without training/data dependencies. */
public final class InferenceConsumer {
    public static void main(String[] args) throws Exception {
        var weights = Files.createTempFile("nnjava-inference-", ".nnj");
        var description = Files.createTempFile("nnjava-definition-", ".nnjd");
        var definition = CausalLanguageModel.builder().name("consumer-decoder").vocabularySize(5)
                .modelWidth(4).blockCount(2).contextLength(4).attention(a -> a.queryHeads(2).keyValueHeads(1))
                .feedForward(f -> f.hiddenWidth(8)).seed(42).define();
        try {
            ModelDefinitions.write(description, definition.specification());
            var restoredDefinition = CausalLanguageModel.Definition.fromSpec(ModelDefinitions.read(description));
            try (var context = ExecutionContext.cpu(); var model = definition.build(context);
                    var restored = restoredDefinition.build(context);
                    var tokens = context.tensor(new int[] {0, 1, 2}, Shape.of(1, 3))) {
                try (var network = NeuralNetwork.builder().inputFeatures(2).dense("projection", 3).build(context);
                        var input = context.tensor(new float[] {1, 2}, Shape.of(1, 2)); var output = network.forward(input)) {
                    if (!output.shape().equals(Shape.of(1, 3))) { throw new AssertionError("Fluent NN shape differs"); }
                }
                model.training(false); StateArchive.saveWeights(weights, model); StateArchive.read(weights).restoreModel(restored);
                try (var expected = model.forward(tokens); var actual = restored.forward(tokens)) {
                    if (!Arrays.equals(expected.toFloatArray(), actual.toFloatArray())) { throw new AssertionError("Restored model differs"); }
                    try (var session = restored.newSession(1); var prefix = Ops.slice(tokens, 1, 0, 2);
                            var suffix = Ops.slice(tokens, 1, 2, 3); var ignored = session.prefill(prefix);
                            var cached = session.decode(suffix); var last = Ops.slice(actual, 1, 2, 3)) {
                        float[] a = cached.toFloatArray(), b = last.toFloatArray();
                        for (int i = 0; i < a.length; i++) {
                            if (Math.abs(a[i] - b[i]) > 2e-5f) { throw new AssertionError("Cached decoder differs"); }
                        }
                    }
                }
                System.out.println("Fluent NN/LM definitions, model-only restore and cached inference passed");
            }
        } finally { Files.deleteIfExists(weights); Files.deleteIfExists(description); }
    }
}

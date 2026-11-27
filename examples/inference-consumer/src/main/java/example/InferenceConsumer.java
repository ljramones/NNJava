package example;

import java.nio.file.Files;
import org.dynamisengine.nn.io.StateArchive;
import org.dynamisengine.nn.nn.Linear;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;

/** A model-only consumer: no training, datasets, compatibility starter, or GPU jars. */
public final class InferenceConsumer {
    public static void main(String[] args) throws Exception {
        var path = Files.createTempFile("nnjava-inference-", ".nnj");
        try (var context = ExecutionContext.cpu(); var model = new Linear(context, 2, 3, 42);
                var restored = new Linear(context, 2, 3, 99);
                var input = context.tensor(new float[] {1, 2}, Shape.of(1, 2))) {
            StateArchive.saveWeights(path, model);
            StateArchive.read(path).restoreModel(restored);
            try (var original = model.forward(input); var loaded = restored.forward(input)) {
                if (!java.util.Arrays.equals(original.toFloatArray(), loaded.toFloatArray())) {
                    throw new AssertionError("Loaded inference model differs");
                }
            }
            System.out.println("Model-only inference and archive restore passed");
        } finally { Files.deleteIfExists(path); }
    }
}

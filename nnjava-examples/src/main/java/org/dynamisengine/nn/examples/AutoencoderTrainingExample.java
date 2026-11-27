package org.dynamisengine.nn.examples;

import java.nio.file.Files;
import java.util.Map;
import org.dynamisengine.nn.data.DataLoader;
import org.dynamisengine.nn.data.TensorDataset;
import org.dynamisengine.nn.nn.Activation;
import org.dynamisengine.nn.nn.NeuralNetwork;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.training.LearningRateSchedule;
import org.dynamisengine.nn.training.Objectives;
import org.dynamisengine.nn.training.Optimizers;
import org.dynamisengine.nn.training.Trainer;

/** A bottleneck autoencoder with shuffled batching and reproducible training resume. */
public final class AutoencoderTrainingExample {
    private AutoencoderTrainingExample() { }

    public static void main(String[] args) throws Exception {
        var definition = NeuralNetwork.builder().name("autoencoder").inputFeatures(3).seed(42)
                .dense("encoder",6,Activation.Kind.TANH).dense("bottleneck",2,Activation.Kind.TANH)
                .dense("decoder",6,Activation.Kind.TANH).dense("reconstruction",3).define();
        var optimizer = Optimizers.adamW().learningRate(.02f)
                .schedule(LearningRateSchedule.warmupCosine(4,400,.1f)).define();
        var directory = Files.createTempDirectory("nnjava-autoencoder-");
        var checkpoint = directory.resolve("training.nnj");
        try {
            try (var context = ExecutionContext.cpu(); var original = definition.build(context); var restored = definition.build(context);
                 var samples = context.tensor(new float[]{-1,-1,-.75f,-1,1,-.25f,1,-1,.25f,1,1,.75f,0,0,0,.5f,-.5f,.125f},Shape.of(6,3));
                 var dataset = new TensorDataset(Map.of("input",samples,"target",samples));
                 var loader = DataLoader.builder(context,dataset).batchSize(3).shuffle(true).seed(24).build();
                 var resumedLoader = DataLoader.builder(context,dataset).batchSize(3).shuffle(true).seed(24).build();
                 var trainer = Trainer.builder().model(original).optimizer(optimizer).objective(Objectives.meanSquaredError()).clipGlobalNorm(1).build();
                 var resumed = Trainer.builder().model(restored).optimizer(optimizer).objective(Objectives.meanSquaredError()).clipGlobalNorm(1).build()) {
                double first = trainer.evaluate(loader).loss();
                trainer.fit(loader,100); trainer.save(checkpoint,loader);
                resumed.restore(checkpoint,resumedLoader);
                trainer.fit(loader,100); resumed.fit(resumedLoader,100);
                double last = resumed.evaluate(resumedLoader).loss();
                try (var expected = original.forward(samples); var actual = restored.forward(samples)) {
                    if (last >= first*.05 || !java.util.Arrays.equals(expected.toFloatArray(),actual.toFloatArray())) {
                        throw new AssertionError("Autoencoder learning or checkpoint resume failed");
                    }
                }
                System.out.printf("Trainer autoencoder MSE: %.6f -> %.6f; %d updates; exact checkpoint resume%n",first,last,resumed.progress().steps());
            }
        } finally { Files.deleteIfExists(checkpoint); Files.deleteIfExists(directory); }
    }
}

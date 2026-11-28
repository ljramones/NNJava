package org.dynamisengine.nn.examples;

import java.nio.file.Files;
import java.util.Arrays;
import java.util.Map;
import java.util.SplittableRandom;
import org.dynamisengine.nn.data.DataLoader;
import org.dynamisengine.nn.data.TensorDataset;
import org.dynamisengine.nn.io.ModelDefinitions;
import org.dynamisengine.nn.nn.Activation;
import org.dynamisengine.nn.nn.Layers;
import org.dynamisengine.nn.nn.NeuralNetwork;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.training.Optimizers;
import org.dynamisengine.nn.training.Trainer;

/** Small LeNet-style classifier for noisy vertical/horizontal bars, using only public APIs. */
public final class CnnTrainingExample {
    private CnnTrainingExample() { }

    public static void main(String[] args) throws Exception {
        var definition=NeuralNetwork.builder().name("bar-classifier").inputShape(1,8,8).seed(42)
                .conv2d("edges",Layers.conv2d(4,3).padding(1).activation(Activation.Kind.RELU)).maxPool2d("max",2)
                .conv2d("features",Layers.conv2d(6,3).padding(1).activation(Activation.Kind.RELU)).avgPool2d("average",2)
                .flatten("flat").dense("hidden",8,Activation.Kind.TANH).dense("logits",2).define();
        var optimizer=Optimizers.adamW().learningRate(.015f).define();
        var directory=Files.createTempDirectory("nnjava-cnn-");
        var description=directory.resolve("model.nnjd"); var checkpoint=directory.resolve("training.nnj");
        try {
            ModelDefinitions.write(description,definition.specification());
            var restoredDefinition=NeuralNetwork.Definition.fromSpec(ModelDefinitions.read(description));
            try (var context=ExecutionContext.cpu(); var model=definition.build(context); var restored=restoredDefinition.build(context);
                 var training=images(context,24,17); var heldOut=images(context,10,97);
                 var loader=DataLoader.builder(context,training).batchSize(8).shuffle(true).seed(24).build();
                 var resumedLoader=DataLoader.builder(context,training).batchSize(8).shuffle(true).seed(24).build();
                 var validation=DataLoader.builder(context,heldOut).batchSize(6).build();
                 var trainer=Trainer.builder().model(model).optimizer(optimizer).clipGlobalNorm(1).build();
                 var resumed=Trainer.builder().model(restored).optimizer(optimizer).clipGlobalNorm(1).build()) {
                double first=trainer.evaluate(validation).loss();
                trainer.fit(loader,40); trainer.save(checkpoint,loader); resumed.restore(checkpoint,resumedLoader);
                trainer.fit(loader,40); resumed.fit(resumedLoader,40);
                var result=resumed.evaluate(validation);
                try (var batch=heldOut.batch(context,new int[]{0,1,2,3}); var input=batch.get("input");
                     var expected=model.forward(input); var actual=restored.forward(input)) {
                    if (!Arrays.equals(expected.toFloatArray(),actual.toFloatArray()) || result.loss()>=first*.15 || result.score()<.9) {
                        throw new AssertionError("CNN learning, validation, or checkpoint resume failed");
                    }
                }
                System.out.printf("Fluent CNN CE: %.6f -> %.6f; held-out accuracy: %.3f; %d updates; exact checkpoint resume%n",
                        first,result.loss(),result.score(),resumed.progress().steps());
            }
        } finally { Files.deleteIfExists(description); Files.deleteIfExists(checkpoint); Files.deleteIfExists(directory); }
    }

    private static TensorDataset images(ExecutionContext context,int count,long seed) {
        float[] images=new float[count*64]; int[] labels=new int[count]; var random=new SplittableRandom(seed);
        for (int image=0;image<count;image++) {
            int label=image%2, position=1+(image/2)%5; labels[image]=label;
            for (int y=0;y<8;y++) for (int x=0;x<8;x++) {
                boolean bar=label==0 ? x>=position && x<position+2 : y>=position && y<position+2;
                images[image*64+y*8+x]=(bar ? 1f : 0f)+(float)(random.nextDouble()-.5)*.12f;
            }
        }
        try (var input=context.tensor(images,Shape.of(count,1,8,8)); var target=context.tensor(labels,Shape.of(count))) {
            return new TensorDataset(Map.of("input",input,"target",target));
        }
    }
}

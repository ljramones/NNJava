package example;

import java.nio.file.Files;
import java.util.Arrays;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import org.dynamisengine.nn.formats.Safetensors;
import org.dynamisengine.nn.io.WeightLoader;
import org.dynamisengine.nn.nn.Linear;
import org.dynamisengine.nn.io.ModelDefinitions;
import org.dynamisengine.nn.io.StateArchive;
import org.dynamisengine.nn.nn.NeuralNetwork;
import org.dynamisengine.nn.nn.Layers;
import org.dynamisengine.nn.nn.Activation;
import org.dynamisengine.nn.nn.Recurrent;
import org.dynamisengine.nn.nn.RecurrentCell;
import org.dynamisengine.nn.nn.RecurrentState;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.transformer.CausalLanguageModel;

/** Fluent model definitions, archives and decoder sessions without training/data dependencies. */
public final class InferenceConsumer {
    public static void main(String[] args) throws Exception {
        verifyExternalWeights();
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
                var cnnDefinition=NeuralNetwork.builder().inputShape(1,5,5).seed(42)
                        .conv2d("features",Layers.conv2d(2,3).padding(1).activation(Activation.Kind.RELU))
                        .maxPool2d("pool",2).flatten("flat").dense("head",2).define();
                ModelDefinitions.write(description,cnnDefinition.specification());
                try (var cnn=cnnDefinition.build(context); var restoredCnn=NeuralNetwork.Definition.fromSpec(ModelDefinitions.read(description)).build(context);
                     var image=context.normal(Shape.of(2,1,5,5),7,.2f,false)) {
                    cnn.training(false); StateArchive.saveWeights(weights,cnn); StateArchive.read(weights).restoreModel(restoredCnn);
                    try (var expected=cnn.forward(image); var actual=restoredCnn.forward(image)) {
                        if (!actual.shape().equals(Shape.of(2,2)) || !Arrays.equals(expected.toFloatArray(),actual.toFloatArray())) {
                            throw new AssertionError("Packaged CNN definition/weights restore failed");
                        }
                    }
                }
                for (var kind:RecurrentCell.Kind.values()) {
                    var recurrentDefinition=Recurrent.builder().kind(kind).inputFeatures(2).hiddenSize(3).seed(42).define();
                    ModelDefinitions.write(description,recurrentDefinition.specification());
                    try (var sequence=recurrentDefinition.build(context);
                         var loaded=Recurrent.Definition.fromSpec(ModelDefinitions.read(description)).build(context);
                         var input=context.normal(Shape.of(2,4,2),7,.2f,false)) {
                        StateArchive.saveWeights(weights,sequence); StateArchive.read(weights).restoreModel(loaded);
                        try (var full=sequence.run(input); var expected=full.output(); var prefix=Ops.slice(input,1,0,2); var suffix=Ops.slice(input,1,2,4);
                             var first=sequence.run(prefix); var carried=first.state(); var state=RecurrentState.fromSnapshot(context,carried.snapshot());
                             var last=loaded.run(suffix,state); var actual=last.output(); var tail=Ops.slice(expected,1,2,4)) {
                            if (!Arrays.equals(tail.toFloatArray(),actual.toFloatArray())) { throw new AssertionError("Packaged "+kind+" state/weight restore failed"); }
                        }
                    }
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
                System.out.println("Fluent NN/CNN/RNN/LM definitions, model-only/state restore and cached inference passed");
            }
        } finally { Files.deleteIfExists(weights); Files.deleteIfExists(description); }
    }

    private static void verifyExternalWeights() throws Exception {
        var path = Files.createTempFile("nnjava-pretrained-", ".safetensors");
        byte[] header = "{\"external.kernel\":{\"dtype\":\"F32\",\"shape\":[2,1],\"data_offsets\":[0,8]}}".getBytes(StandardCharsets.UTF_8);
        try {
            Files.write(path, ByteBuffer.allocate(8 + header.length + 8).order(ByteOrder.LITTLE_ENDIAN)
                    .putLong(header.length).put(header).putFloat(2).putFloat(3).array());
            try (var context = ExecutionContext.cpu(); var model = Linear.builder().inputFeatures(2).outputFeatures(1).bias(false).build(context);
                 var file = Safetensors.open(path)) {
                WeightLoader.builder().transpose("external.kernel", "weight").build().load(file, model);
                try (var input = context.tensor(new float[]{4, 5}, Shape.of(1, 2)); var output = model.forward(input)) {
                    if (output.toFloatArray()[0] != 23) { throw new AssertionError("Packaged safetensors weight mapping failed"); }
                }
            }
        } finally { Files.deleteIfExists(path); }
    }
}

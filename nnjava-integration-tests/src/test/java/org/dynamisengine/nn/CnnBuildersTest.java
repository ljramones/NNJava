package org.dynamisengine.nn;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import org.dynamisengine.nn.backend.Backend;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.backend.vectrix.VectrixBackend;
import org.dynamisengine.nn.data.DataLoader;
import org.dynamisengine.nn.data.TensorDataset;
import org.dynamisengine.nn.io.ModelDefinitions;
import org.dynamisengine.nn.io.StateArchive;
import org.dynamisengine.nn.model.ComponentSpec;
import org.dynamisengine.nn.nn.*;
import org.dynamisengine.nn.ops.Conv2dConfig;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.ops.Pool2dConfig;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.training.Optimizers;
import org.dynamisengine.nn.training.Trainer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class CnnBuildersTest {
    @TempDir Path temporary;

    private static NeuralNetwork.Definition definition() {
        return NeuralNetwork.builder().name("cnn").inputShape(2,7,9).seed(42)
                .conv2d("features",Layers.conv2d(4,2,3).stride(2,1).padding(1).dilation(1,2).groups(2).activation(Activation.Kind.RELU))
                .maxPool2d("max",2).layer("average",Layers.avgPool2d(2).stride(1)).flatten("flat").dense("head",3).define();
    }

    @Test void cnnDefinitionsInferShapesAndRoundTripWithoutAllocatingWeights() throws Exception {
        var definition=definition();
        assertEquals(Shape.of(3),definition.outputShape()); assertEquals(Shape.of(5,3),definition.outputShape(Shape.of(5,2,7,9)));
        Path path=temporary.resolve("cnn.nnjd"); ModelDefinitions.write(path,definition.specification());
        assertEquals(definition,NeuralNetwork.Definition.fromSpec(ModelDefinitions.read(path)));
        try (var context=ExecutionContext.cpu()) { definition.validate(context); assertEquals(0,context.liveAllocations()); }
    }

    @Test void directConvBuilderSnapshotsConfigurationAndMatchesFunctionalComposition() {
        var builder=Conv2d.builder().name("projection").inputChannels(4).outputChannels(8).kernelSize(2,3)
                .stride(2,1).padding(1).dilation(1,2).groups(2).bias(false).seed(42);
        var definition=builder.define(); builder.outputChannels(4).seed(8);
        assertEquals(8,definition.outputChannels()); assertEquals(42,definition.seed());
        assertEquals(definition,Conv2d.Definition.fromSpec(definition.specification()));
        try (var context=ExecutionContext.cpu(); var layer=definition.build(context); var input=context.normal(Shape.of(2,4,7,9),5,.2f,false);
             var weights=layer.namedParameters().get("weight").value(); var actual=layer.forward(input); var expected=Ops.conv2d(input,weights,definition.config())) {
            assertArrayEquals(expected.toFloatArray(),actual.toFloatArray()); assertEquals(1,layer.namedParameters().size());
        }
    }

    @Test void fluentConvolutionUsesStableNamedSeedsAndMatchesDirectComposition() {
        long seed=19;
        var definition=NeuralNetwork.builder().inputShape(2,4,5).seed(seed)
                .conv2d("features",Layers.conv2d(4,3).padding(1).activation(Activation.Kind.TANH)).maxPool2d("pool",2).define();
        try (var context=ExecutionContext.cpu(); var model=definition.build(context);
             var direct=new Sequential(context,new Conv2d(context,2,4,new Conv2dConfig(3,3).padding(1),ModelBuild.seed(seed,"features")),
                     Activation.tanh(context),new Pool2d(context,Pool2d.Kind.MAX,new Pool2dConfig(2)));
             var input=context.normal(Shape.of(3,2,4,5),8,.3f,false); var actual=model.forward(input); var expected=direct.forward(input)) {
            assertArrayEquals(expected.toFloatArray(),actual.toFloatArray());
            assertTrue(model.namedParameters().containsKey("features.conv.weight")); assertTrue(model.namedParameters().containsKey("features.conv.bias"));
        }
    }

    @Test void cnnModelOnlyDefinitionAndWeightsRestorePredictions() throws Exception {
        var definition=definition(); Path description=temporary.resolve("cnn.nnjd"), weights=temporary.resolve("weights.nnj");
        ModelDefinitions.write(description,definition.specification());
        try (var context=ExecutionContext.cpu(); var model=definition.build(context);
             var input=context.normal(Shape.of(2,2,7,9),2,.2f,false)) {
            model.training(false); StateArchive.saveWeights(weights,model);
            try (var restored=NeuralNetwork.Definition.fromSpec(ModelDefinitions.read(description)).build(context)) {
                StateArchive.read(weights).restoreModel(restored);
                try (var actual=restored.forward(input); var expected=model.forward(input)) { assertArrayEquals(expected.toFloatArray(),actual.toFloatArray()); }
            }
        }
    }

    @Test void invalidShapesGroupsAndFutureSchemasFailBeforeAllocating() {
        assertThrows(IllegalArgumentException.class, () -> NeuralNetwork.builder().inputShape(1,2,2).conv2d("bad",4,3).define());
        assertThrows(IllegalArgumentException.class, () -> NeuralNetwork.builder().inputFeatures(8).maxPool2d("bad",2).define());
        assertThrows(IllegalArgumentException.class, () -> NeuralNetwork.builder().inputShape(3,5,5).conv2d("bad",Layers.conv2d(4,3).groups(2)).define());
        assertThrows(IllegalArgumentException.class, () -> Conv2d.builder().inputChannels(3).outputChannels(4).groups(2).define());
        var spec=definition().specification(); var children=new ArrayList<>(spec.children()); var child=children.getFirst();
        children.set(0,new ComponentSpec(child.type(),99,child.name(),child.attributes(),child.children()));
        assertThrows(IllegalArgumentException.class, () -> NeuralNetwork.Definition.fromSpec(new ComponentSpec(spec.type(),1,spec.name(),spec.attributes(),children)));
        var attrs=new LinkedHashMap<>(child.attributes()); attrs.put("unsupported","true");
        children.set(0,new ComponentSpec(child.type(),1,child.name(),attrs,child.children()));
        assertThrows(IllegalArgumentException.class, () -> NeuralNetwork.Definition.fromSpec(new ComponentSpec(spec.type(),1,spec.name(),spec.attributes(),children)));
    }

    @Test void unsupportedPoolingBackwardIsReportedBeforeAnyNetworkAllocation() {
        var allocations=new AtomicInteger();
        try (var context=ExecutionContext.create(backend(allocations,-1,Operation.MAX_POOL2D_BACKWARD))) {
            var definition=NeuralNetwork.builder().name("cnn").inputShape(1,5,5).conv2d("features",4,3).maxPool2d("pool",2).define();
            var error=assertThrows(UnsupportedOperationException.class, () -> definition.build(context));
            assertTrue(error.getMessage().contains("cnn.pool")); assertEquals(0,allocations.get()); assertEquals(0,context.liveAllocations());
        }
    }

    @Test void everyPartialCnnParameterAllocationIsReleasedOnConstructionFailure() {
        var definition=NeuralNetwork.builder().inputShape(1,5,5).conv2d("features",2,3).flatten("flat").dense("head",2).define();
        for (int failure=1;failure<=4;failure++) {
            var allocations=new AtomicInteger();
            try (var context=ExecutionContext.create(backend(allocations,failure,null))) {
                assertThrows(IllegalStateException.class, () -> definition.build(context)); assertEquals(0,context.liveAllocations());
                try (var usable=context.full(Shape.of(),2)) { assertEquals(2,usable.item()); }
            }
        }
    }

    @Test void cnnTrainerResumesMidEpochWithIdenticalFinalCheckpointBytes() throws Exception {
        var definition=NeuralNetwork.builder().inputShape(1,4,4).seed(42).conv2d("features",2,2,Activation.Kind.TANH)
                .avgPool2d("pool",2).flatten("flat").dense("head",2).define();
        var optimizer=Optimizers.adamW().learningRate(.02f).define(); Path interrupted=temporary.resolve("partial.nnj"), a=temporary.resolve("a.nnj"), b=temporary.resolve("b.nnj");
        try (var context=ExecutionContext.cpu(); var original=definition.build(context); var restored=definition.build(context);
             var input=context.normal(Shape.of(3,1,4,4),9,.2f,false); var target=context.tensor(new int[]{0,1,0},Shape.of(3));
             var dataset=new TensorDataset(Map.of("input",input,"target",target));
             var loader=DataLoader.builder(context,dataset).batchSize(2).shuffle(true).seed(7).build();
             var restoredLoader=DataLoader.builder(context,dataset).batchSize(2).shuffle(true).seed(7).build();
             var first=Trainer.builder().model(original).optimizer(optimizer).build(); var second=Trainer.builder().model(restored).optimizer(optimizer).build()) {
            first.fit(loader,null,4,1); first.save(interrupted,loader); second.restore(interrupted,restoredLoader);
            assertEquals(first.fit(loader,4),second.fit(restoredLoader,4));
            first.save(a,loader); second.save(b,restoredLoader);
            assertArrayEquals(java.nio.file.Files.readAllBytes(a),java.nio.file.Files.readAllBytes(b));
        }
    }

    private static Backend backend(AtomicInteger allocations,int failAt,Operation unsupported) {
        var delegate=new VectrixBackend();
        return (Backend)Proxy.newProxyInstance(Backend.class.getClassLoader(),new Class<?>[]{Backend.class},(proxy,method,args) -> {
            if (method.getName().equals("allocate") && allocations.incrementAndGet()==failAt) { throw new IllegalStateException("Injected allocation failure"); }
            if (method.getName().equals("supports") && args.length==2 && args[0]==unsupported) { return false; }
            try { return method.invoke(delegate,args); } catch (InvocationTargetException error) { throw error.getCause(); }
        });
    }
}

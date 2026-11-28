package org.dynamisengine.nn;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.nio.file.Files;
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
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.training.LearningRateSchedule;
import org.dynamisengine.nn.training.Optimizers;
import org.dynamisengine.nn.training.Trainer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class RecurrentBuildersTest {
    @TempDir Path temporary;

    private static Layers.RecurrentSpec specification(RecurrentCell.Kind kind,long hidden) {
        return switch (kind) { case RNN -> Layers.rnn(hidden); case LSTM -> Layers.lstm(hidden); case GRU -> Layers.gru(hidden); };
    }

    @Test void cellSequenceAndNetworkDefinitionsSnapshotAndRoundTripWithoutAllocation() throws Exception {
        for (var kind:RecurrentCell.Kind.values()) {
            var builder=RecurrentCell.builder().kind(kind).inputFeatures(2).hiddenSize(3).bias(false).seed(19);
            var cell=builder.define(); builder.hiddenSize(5); assertEquals(3,cell.hiddenSize());
            assertEquals(cell,RecurrentCell.Definition.fromSpec(cell.specification()));
            var sequence=Recurrent.builder().kind(kind).inputFeatures(2).hiddenSize(3).output(SequenceOutput.LAST).seed(19).define();
            assertEquals(sequence,Recurrent.Definition.fromSpec(sequence.specification()));
            var definition=NeuralNetwork.builder().name("sequence-classifier").inputShape(5,2).seed(42)
                    .layer("memory",specification(kind,3).output(SequenceOutput.LAST)).dense("head",2).define();
            Path path=temporary.resolve(kind+".nnjd"); ModelDefinitions.write(path,definition.specification());
            assertEquals(definition,NeuralNetwork.Definition.fromSpec(ModelDefinitions.read(path)));
            assertEquals(Shape.of(2),definition.outputShape()); assertEquals(Shape.of(7,2),definition.outputShape(Shape.of(7,5,2)));
            try (var context=ExecutionContext.cpu()) { cell.validate(context); sequence.validate(context); definition.validate(context); assertEquals(0,context.liveAllocations()); }
        }
    }

    @Test void fluentStackedRnnLstmAndGruLayersInferShapesAndReuseDirectComponentSeeds() {
        long seed=19;
        var definition=NeuralNetwork.builder().inputShape(4,2).seed(seed).rnn("rnn",4).lstm("lstm",3).gru("gru",2,SequenceOutput.LAST).dense("head",2).define();
        try (var context=ExecutionContext.cpu(); var model=definition.build(context);
             var direct=new Sequential(context,
                     Recurrent.builder().kind(RecurrentCell.Kind.RNN).inputFeatures(2).hiddenSize(4).seed(ModelBuild.seed(seed,"rnn")).build(context),
                     Recurrent.builder().kind(RecurrentCell.Kind.LSTM).inputFeatures(4).hiddenSize(3).seed(ModelBuild.seed(seed,"lstm")).build(context),
                     Recurrent.builder().kind(RecurrentCell.Kind.GRU).inputFeatures(3).hiddenSize(2).output(SequenceOutput.LAST).seed(ModelBuild.seed(seed,"gru")).build(context),
                     new Linear(context,2,2,ModelBuild.seed(seed,"head")));
             var input=context.tensor(RecurrentCellTest.values(16),Shape.of(2,4,2)); var actual=model.forward(input); var expected=direct.forward(input)) {
            assertArrayEquals(expected.toFloatArray(),actual.toFloatArray());
            assertTrue(model.namedParameters().containsKey("lstm.cell.input.weight"));
        }
    }

    @Test void insertingParameterlessLayersDoesNotChangeNamedRecurrentInitialization() {
        var a=NeuralNetwork.builder().inputShape(3,2).seed(42).lstm("memory",4,SequenceOutput.LAST).dense("head",2).define();
        var b=NeuralNetwork.builder().inputShape(3,2).seed(42).dropout("off",0).lstm("memory",4,SequenceOutput.LAST).dense("head",2).define();
        try (var context=ExecutionContext.cpu(); var first=a.build(context); var second=b.build(context)) {
            first.namedParameters().forEach((name,parameter) -> {
                try (var x=parameter.value(); var y=second.namedParameters().get(name).value()) { assertArrayEquals(x.toFloatArray(),y.toFloatArray()); }
            });
        }
    }

    @Test void modelOnlyArchivesRestoreEachCellKindAndRejectIncompatibleConfiguration() throws Exception {
        for (var kind:RecurrentCell.Kind.values()) {
            var definition=Recurrent.builder().kind(kind).inputFeatures(2).hiddenSize(3).seed(42).define();
            Path path=temporary.resolve(kind+".nnj");
            try (var context=ExecutionContext.cpu(); var model=definition.build(context); var restored=definition.build(context);
                 var incompatible=Recurrent.builder().kind(kind).inputFeatures(2).hiddenSize(3).output(SequenceOutput.LAST).seed(42).build(context);
                 var input=context.tensor(RecurrentCellTest.values(12),Shape.of(2,3,2))) {
                model.training(false); StateArchive.saveWeights(path,model); StateArchive.read(path).restoreModel(restored);
                try (var expected=model.forward(input); var actual=restored.forward(input)) { assertArrayEquals(expected.toFloatArray(),actual.toFloatArray()); }
                assertThrows(IllegalArgumentException.class, () -> StateArchive.read(path).restoreModel(incompatible));
            }
        }
    }

    @Test void fullTrainerCheckpointsResumeEveryRecurrentKindWithIdenticalFinalArchiveBytes() throws Exception {
        for (var kind:RecurrentCell.Kind.values()) {
            var definition=NeuralNetwork.builder().inputShape(3,1).seed(42).layer("memory",specification(kind,3).output(SequenceOutput.LAST))
                    .dropout("regularization",.15f).dense("head",2).define();
            var optimizer=Optimizers.adamW().learningRate(.02f).schedule(LearningRateSchedule.warmupCosine(2,20,.1f)).define();
            Path checkpoint=temporary.resolve(kind+"-partial.nnj"), a=temporary.resolve(kind+"-a.nnj"), b=temporary.resolve(kind+"-b.nnj");
            try (var context=ExecutionContext.cpu(); var original=definition.build(context); var restored=definition.build(context);
                 var input=context.tensor(RecurrentCellTest.values(15),Shape.of(5,3,1)); var target=context.tensor(new int[]{0,1,0,1,0},Shape.of(5));
                 var dataset=new TensorDataset(Map.of("input",input,"target",target));
                 var loader=DataLoader.builder(context,dataset).batchSize(2).shuffle(true).seed(7).build();
                 var restoredLoader=DataLoader.builder(context,dataset).batchSize(2).shuffle(true).seed(7).build();
                 var first=Trainer.builder().model(original).optimizer(optimizer).clipGlobalNorm(.5).build();
                 var second=Trainer.builder().model(restored).optimizer(optimizer).clipGlobalNorm(.5).build()) {
                first.fit(loader,null,3,2); first.save(checkpoint,loader); second.restore(checkpoint,restoredLoader);
                assertEquals(first.fit(loader,3),second.fit(restoredLoader,3));
                first.save(a,loader); second.save(b,restoredLoader); assertArrayEquals(Files.readAllBytes(a),Files.readAllBytes(b));
            }
        }
    }

    @Test void invalidDefinitionsUnknownSettingsAndFutureSchemasFailWithoutResources() {
        assertThrows(IllegalArgumentException.class, () -> Layers.lstm(0));
        assertThrows(ArithmeticException.class, () -> Layers.lstm(Long.MAX_VALUE));
        assertThrows(IllegalArgumentException.class, () -> RecurrentCell.builder().inputFeatures(0).hiddenSize(3).define());
        assertThrows(IllegalArgumentException.class, () -> NeuralNetwork.builder().inputFeatures(3).lstm("bad",4).define());
        var original=NeuralNetwork.builder().inputShape(3,2).lstm("memory",4).define().specification();
        var children=new ArrayList<>(original.children()); var child=children.getFirst();
        children.set(0,new ComponentSpec(child.type(),99,child.name(),child.attributes(),child.children()));
        assertThrows(IllegalArgumentException.class, () -> NeuralNetwork.Definition.fromSpec(new ComponentSpec(original.type(),1,original.name(),original.attributes(),children)));
        var attrs=new LinkedHashMap<>(child.attributes()); attrs.put("bidirectional","true");
        children.set(0,new ComponentSpec(child.type(),1,child.name(),attrs,child.children()));
        assertThrows(IllegalArgumentException.class, () -> NeuralNetwork.Definition.fromSpec(new ComponentSpec(original.type(),1,original.name(),original.attributes(),children)));
    }

    @Test void missingBackwardAndMaskCapabilitiesAreReportedBeforeNetworkAllocation() {
        for (var unavailable:new Operation[]{Operation.SIGMOID_BACKWARD,Operation.WHERE}) {
            var allocations=new AtomicInteger();
            try (var context=ExecutionContext.create(backend(allocations,new AtomicInteger(-1),unavailable))) {
                var definition=NeuralNetwork.builder().name("classifier").inputShape(3,2).lstm("memory",4,SequenceOutput.LAST).dense("head",2).define();
                var error=assertThrows(UnsupportedOperationException.class, () -> definition.build(context));
                assertTrue(error.getMessage().contains("classifier.memory")); assertEquals(0,allocations.get()); assertEquals(0,context.liveAllocations());
            }
        }
    }

    @Test void everyPartialNetworkConstructionAllocationIsReleasedForAllCellKinds() {
        for (var kind:RecurrentCell.Kind.values()) for (int fail=1;fail<=6;fail++) {
            var allocations=new AtomicInteger();
            try (var context=ExecutionContext.create(backend(allocations,new AtomicInteger(fail),null))) {
                var definition=NeuralNetwork.builder().inputShape(3,2).layer("memory",specification(kind,3).output(SequenceOutput.LAST)).dense("head",2).define();
                assertThrows(IllegalStateException.class, () -> definition.build(context)); assertEquals(0,context.liveAllocations());
                try (var usable=context.full(Shape.of(),3)) { assertEquals(3,usable.item()); }
            }
        }
    }

    @Test void everyForwardAllocationFailureReleasesTransientStatesAndAllowsRetry() {
        int count;
        var allocations=new AtomicInteger(); var failure=new AtomicInteger(-1);
        try (var context=ExecutionContext.create(backend(allocations,failure,null));
             var model=Recurrent.builder().kind(RecurrentCell.Kind.LSTM).inputFeatures(2).hiddenSize(3).build(context);
             var input=context.tensor(RecurrentCellTest.values(8),Shape.of(2,2,2))) {
            int before=allocations.get(); try (var result=model.run(input)) { }
            count=allocations.get()-before;
        }
        assertTrue(count>20);
        for (int fail=1;fail<=count;fail++) {
            allocations.set(0); failure.set(-1);
            try (var context=ExecutionContext.create(backend(allocations,failure,null));
                 var model=Recurrent.builder().kind(RecurrentCell.Kind.LSTM).inputFeatures(2).hiddenSize(3).build(context);
                 var input=context.tensor(RecurrentCellTest.values(8),Shape.of(2,2,2))) {
                int before=context.liveAllocations(); failure.set(allocations.get()+fail);
                assertThrows(IllegalStateException.class, () -> model.run(input)); assertEquals(before,context.liveAllocations(),"allocation "+fail);
                try (var result=model.run(input); var output=result.output()) { assertEquals(Shape.of(2,2,3),output.shape()); }
            }
        }
    }

    private static Backend backend(AtomicInteger allocations,AtomicInteger failAt,Operation unavailable) {
        var delegate=new VectrixBackend();
        return (Backend)Proxy.newProxyInstance(Backend.class.getClassLoader(),new Class<?>[]{Backend.class},(proxy,method,args) -> {
            if (method.getName().equals("allocate") && allocations.incrementAndGet()==failAt.get()) { throw new IllegalStateException("Injected allocation failure"); }
            if (method.getName().equals("supports") && args.length==2 && args[0]==unavailable) { return false; }
            try { return method.invoke(delegate,args); } catch (InvocationTargetException error) { throw error.getCause(); }
        });
    }
}

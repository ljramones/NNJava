package org.dynamisengine.nn;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;
import org.dynamisengine.nn.io.StateArchive;
import org.dynamisengine.nn.nn.Recurrent;
import org.dynamisengine.nn.nn.RecurrentCell;
import org.dynamisengine.nn.nn.RecurrentState;
import org.dynamisengine.nn.nn.SequenceOutput;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.state.TensorSnapshot;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class RecurrentSequenceTest {
    @TempDir Path temporary;
    private static Recurrent.Definition definition(RecurrentCell.Kind kind,SequenceOutput output) {
        return Recurrent.builder().kind(kind).inputFeatures(2).hiddenSize(3).output(output).seed(42).define();
    }

    @Test void maskedSequencesMatchIndependentRowUnrollingAndZeroPaddedInputGradients() {
        for (var kind:RecurrentCell.Kind.values()) {
            float[] data=RecurrentCellTest.values(24); int[] lengths={4,1,0};
            for (int row=0;row<3;row++) for (int step=lengths[row];step<4;step++) { Arrays.fill(data,(row*4+step)*2,(row*4+step+1)*2,Float.NaN); }
            float[] initialH=RecurrentCellTest.values(9), initialC=kind==RecurrentCell.Kind.LSTM ? new float[]{.1f,.2f,.3f,.2f,.1f,-.1f,.3f,.4f,.5f} : null;
            try (var context=ExecutionContext.cpu(); var model=definition(kind,SequenceOutput.ALL).build(context);
                 var x=context.tensor(data,Shape.of(3,4,2),true); var h=context.tensor(initialH,Shape.of(3,3),true);
                 var c=initialC==null ? null : context.tensor(initialC,h.shape(),true);
                 var initial=c==null ? RecurrentState.hidden(h) : RecurrentState.lstm(h,c);
                 var valid=context.tensor(lengths,Shape.of(3))) {
                float[] expected=new float[36], finalH=new float[9], finalC=initialC==null ? null : new float[9];
                var weights=RecurrentCellTest.weights(model.cell());
                for (int row=0;row<3;row++) {
                    float[] currentH=Arrays.copyOfRange(initialH,row*3,(row+1)*3), currentC=initialC==null ? null : Arrays.copyOfRange(initialC,row*3,(row+1)*3);
                    for (int step=0;step<lengths[row];step++) {
                        float[] frame=Arrays.copyOfRange(data,(row*4+step)*2,(row*4+step+1)*2);
                        var next=RecurrentCellTest.reference(kind,frame,currentH,currentC,weights,1,2,3);
                        currentH=next[0]; currentC=next[1]; System.arraycopy(currentH,0,expected,(row*4+step)*3,3);
                    }
                    System.arraycopy(currentH,0,finalH,row*3,3);
                    if (finalC!=null) { System.arraycopy(currentC,0,finalC,row*3,3); }
                }
                try (var tape=context.gradientTape(); var result=model.run(x,initial,valid); var output=result.output(); var finalState=result.state();
                     var actualH=finalState.hidden(); var actualC=finalC==null ? null : finalState.cell(); var loss=Ops.sum(output)) {
                    assertArrayEquals(expected,output.toFloatArray(),3e-6f); assertArrayEquals(finalH,actualH.toFloatArray(),3e-6f);
                    if (actualC!=null) { assertArrayEquals(finalC,actualC.toFloatArray(),3e-6f); }
                    tape.backward(loss);
                }
                try (var gradient=x.gradient()) {
                    float[] values=gradient.toFloatArray();
                    for (int row=0;row<3;row++) for (int step=lengths[row];step<4;step++) for (int f=0;f<2;f++) { assertEquals(0,values[(row*4+step)*2+f]); }
                    for (float value:values) { assertTrue(Float.isFinite(value)); }
                }
                model.namedParameters().values().forEach(parameter -> {
                    try (var value=parameter.value(); var gradient=value.gradient()) { for (float component:gradient.toFloatArray()) { assertTrue(Float.isFinite(component)); } }
                });
            }
        }
    }

    @Test void lastOutputIsTheLastValidHiddenStateIncludingZeroLengthRows() {
        for (var kind:RecurrentCell.Kind.values()) {
            try (var context=ExecutionContext.cpu(); var model=definition(kind,SequenceOutput.LAST).build(context);
                 var input=context.tensor(RecurrentCellTest.values(12),Shape.of(2,3,2)); var lengths=context.tensor(new int[]{3,0},Shape.of(2));
                 var h=context.tensor(new float[]{.2f,.3f,.4f,.5f,.6f,.7f},Shape.of(2,3));
                 var c=kind==RecurrentCell.Kind.LSTM ? context.tensor(RecurrentCellTest.values(6),h.shape()) : null;
                 var initial=c==null ? RecurrentState.hidden(h) : RecurrentState.lstm(h,c);
                 var result=model.run(input,initial,lengths); var output=result.output(); var state=result.state(); var hidden=state.hidden()) {
                assertEquals(Shape.of(2,3),output.shape()); assertArrayEquals(hidden.toFloatArray(),output.toFloatArray());
                assertArrayEquals(new float[]{.5f,.6f,.7f},Arrays.copyOfRange(output.toFloatArray(),3,6));
            }
        }
    }

    @Test void chunkedExecutionMatchesFullExecutionForEveryCellKind() {
        for (var kind:RecurrentCell.Kind.values()) {
            try (var context=ExecutionContext.cpu(); var model=definition(kind,SequenceOutput.ALL).build(context);
                 var input=context.tensor(RecurrentCellTest.values(16),Shape.of(2,4,2)); var full=model.run(input);
                 var prefix=Ops.slice(input,1,0,2); var suffix=Ops.slice(input,1,2,4); var first=model.run(prefix); var carried=first.state();
                 var second=model.run(suffix,carried); var a=first.output(); var b=second.output(); var joined=Ops.concat(1,a,b); var expected=full.output();
                 var expectedState=full.state(); var actualState=second.state()) {
                assertArrayEquals(expected.toFloatArray(),joined.toFloatArray()); sameState(expectedState,actualState);
                try (var repeated=model.run(input); var value=repeated.output()) { assertArrayEquals(expected.toFloatArray(),value.toFloatArray()); }
            }
        }
    }

    @Test void fullBpttAcrossChunksMatchesTheUnchunkedInputAndParameterGradients() {
        for (var kind:RecurrentCell.Kind.values()) {
            try (var context=ExecutionContext.cpu(); var a=definition(kind,SequenceOutput.ALL).build(context); var b=definition(kind,SequenceOutput.ALL).build(context);
                 var x=context.tensor(RecurrentCellTest.values(8),Shape.of(1,4,2),true); var y=context.tensor(x.toFloatArray(),x.shape(),true)) {
                try (var tape=context.gradientTape(); var result=a.run(x); var output=result.output(); var loss=Ops.sum(output)) { tape.backward(loss); }
                try (var tape=context.gradientTape(); var prefix=Ops.slice(y,1,0,2); var suffix=Ops.slice(y,1,2,4);
                     var first=b.run(prefix); var state=first.state(); var second=b.run(suffix,state); var left=first.output(); var right=second.output();
                     var joined=Ops.concat(1,left,right); var loss=Ops.sum(joined)) { tape.backward(loss); }
                try (var gx=x.gradient(); var gy=y.gradient()) { assertArrayEquals(gx.toFloatArray(),gy.toFloatArray(),2e-6f); }
                a.namedParameters().forEach((name,parameter) -> {
                    try (var value=parameter.value(); var other=b.namedParameters().get(name).value(); var gx=value.gradient(); var gy=other.gradient()) {
                        assertArrayEquals(gx.toFloatArray(),gy.toFloatArray(),2e-6f,name);
                    }
                });
            }
        }
    }

    @Test void detachingCarriedStateStopsGradientsAtTheChunkBoundary() {
        for (var kind:RecurrentCell.Kind.values()) {
            try (var context=ExecutionContext.cpu(); var model=definition(kind,SequenceOutput.ALL).build(context);
                 var input=context.tensor(RecurrentCellTest.values(8),Shape.of(1,4,2),true)) {
                try (var tape=context.gradientTape(); var prefix=Ops.slice(input,1,0,2); var suffix=Ops.slice(input,1,2,4);
                     var first=model.run(prefix); var state=first.state(); var detached=state.detach(); var second=model.run(suffix,detached);
                     var output=second.output(); var loss=Ops.sum(output)) { tape.backward(loss); }
                try (var gradient=input.gradient()) {
                    float[] values=gradient.toFloatArray(); assertArrayEquals(new float[4],Arrays.copyOfRange(values,0,4));
                    assertTrue(Arrays.stream(new double[]{values[4],values[5],values[6],values[7]}).anyMatch(value -> Math.abs(value)>1e-6));
                }
            }
        }
    }

    @Test void carryingStateAcrossTapesRequiresDetach() {
        try (var context=ExecutionContext.cpu(); var model=definition(RecurrentCell.Kind.LSTM,SequenceOutput.ALL).build(context);
             var input=context.tensor(RecurrentCellTest.values(4),Shape.of(1,2,2),true)) {
            RecurrentState carried;
            try (var tape=context.gradientTape(); var result=model.run(input)) { carried=result.state(); }
            try (carried) {
                try (var tape=context.gradientTape()) { assertThrows(IllegalStateException.class, () -> model.run(input,carried)); }
                try (var detached=carried.detach(); var tape=context.gradientTape(); var result=model.run(input,detached); var output=result.output(); var loss=Ops.sum(output)) {
                    tape.backward(loss);
                }
            }
        }
    }

    @Test void rowResetsZeroOnlySelectedStatesAndTheirGradientHistory() {
        try (var context=ExecutionContext.cpu(); var h=context.tensor(RecurrentCellTest.values(6),Shape.of(2,3),true);
             var c=context.tensor(new float[]{1,2,3,4,5,6},Shape.of(2,3),true); var mask=context.tensor(new boolean[]{true,false},Shape.of(2));
             var initial=RecurrentState.lstm(h,c); var tape=context.gradientTape(); var reset=initial.resetRows(mask);
             var actualH=reset.hidden(); var actualC=reset.cell(); var hs=Ops.sum(actualH); var cs=Ops.sum(actualC); var loss=Ops.add(hs,cs)) {
            assertArrayEquals(new float[]{0,0,0,4,5,6},actualC.toFloatArray());
            try (var original=initial.hidden()) { assertArrayEquals(h.toFloatArray(),original.toFloatArray()); }
            tape.backward(loss);
            try (var gh=h.gradient(); var gc=c.gradient()) { assertArrayEquals(new float[]{0,0,0,1,1,1},gh.toFloatArray()); assertArrayEquals(gh.toFloatArray(),gc.toFloatArray()); }
        }
    }

    @Test void carriedStateSnapshotsRestoreWithWeightsInANewContext() throws Exception {
        Path weights=temporary.resolve("recurrent.nnj"); var definition=definition(RecurrentCell.Kind.LSTM,SequenceOutput.ALL);
        Map<String,TensorSnapshot> snapshot; float[] expected;
        try (var context=ExecutionContext.cpu(); var model=definition.build(context); var input=context.tensor(RecurrentCellTest.values(8),Shape.of(1,4,2));
             var prefix=Ops.slice(input,1,0,2); var suffix=Ops.slice(input,1,2,4); var first=model.run(prefix); var state=first.state();
             var last=model.run(suffix,state); var output=last.output()) {
            snapshot=state.snapshot(); expected=output.toFloatArray(); StateArchive.saveWeights(weights,model);
        }
        try (var context=ExecutionContext.cpu(); var model=definition.build(context); var initial=RecurrentState.fromSnapshot(context,snapshot);
             var suffix=context.tensor(Arrays.copyOfRange(RecurrentCellTest.values(8),4,8),Shape.of(1,2,2))) {
            StateArchive.read(weights).restoreModel(model);
            try (var result=model.run(suffix,initial); var output=result.output()) { assertArrayEquals(expected,output.toFloatArray()); }
        }
    }

    @Test void unbatchedAndMultipleBatchPrefixesPreserveTheirShapes() {
        for (var output:SequenceOutput.values()) {
            try (var context=ExecutionContext.cpu(); var model=definition(RecurrentCell.Kind.GRU,output).build(context);
                 var input=context.tensor(RecurrentCellTest.values(48),Shape.of(2,3,4,2)); var flat=Ops.reshape(input,Shape.of(6,4,2));
                 var actual=model.forward(input); var expected=model.forward(flat);
                 var single=context.tensor(RecurrentCellTest.values(8),Shape.of(4,2)); var one=model.forward(single)) {
                assertArrayEquals(expected.toFloatArray(),actual.toFloatArray());
                assertEquals(output==SequenceOutput.ALL ? Shape.of(2,3,4,3) : Shape.of(2,3,3),actual.shape());
                assertEquals(output==SequenceOutput.ALL ? Shape.of(4,3) : Shape.of(3),one.shape());
            }
        }
    }

    @Test void stridedSequenceViewsProduceTheSameValuesAndInputGradients() {
        try (var context=ExecutionContext.cpu(); var model=definition(RecurrentCell.Kind.GRU,SequenceOutput.ALL).build(context);
             var base=context.tensor(RecurrentCellTest.values(16),Shape.of(2,2,4),true)) {
            float[] values; try (var view=Ops.transpose(base,1,2)) { values=view.toFloatArray(); }
            try (var direct=context.tensor(values,Shape.of(2,4,2),true)) {
                try (var tape=context.gradientTape(); var input=Ops.transpose(base,1,2); var output=model.forward(input); var loss=Ops.sum(output)) { tape.backward(loss); }
                model.namedParameters().values().forEach(parameter -> parameter.clearGradient());
                try (var tape=context.gradientTape(); var output=model.forward(direct); var loss=Ops.sum(output)) { tape.backward(loss); }
                try (var actual=base.gradient(); var transposed=Ops.transpose(actual,1,2); var expected=direct.gradient()) { assertArrayEquals(expected.toFloatArray(),transposed.toFloatArray()); }
            }
        }
    }

    @Test void badLengthsTypesStatesAndContextsAreRejectedBeforeExecution() {
        try (var context=ExecutionContext.cpu(); var other=ExecutionContext.cpu(); var model=definition(RecurrentCell.Kind.LSTM,SequenceOutput.ALL).build(context);
             var input=context.tensor(RecurrentCellTest.values(12),Shape.of(2,3,2)); var integers=context.tensor(new int[12],input.shape());
             var longLengths=context.tensor(new int[]{4,1},Shape.of(2)); var negative=context.tensor(new int[]{-1,1},Shape.of(2));
             var wrongShape=context.tensor(new int[]{2},Shape.of(1)); var hidden=context.full(Shape.of(2,3),0); var wrongState=RecurrentState.hidden(hidden);
             var foreign=other.tensor(RecurrentCellTest.values(12),input.shape()); var empty=context.tensor(new float[0],Shape.of(2,0,2))) {
            int before=context.liveAllocations();
            assertThrows(IllegalArgumentException.class, () -> model.run(input,null,longLengths));
            assertThrows(IllegalArgumentException.class, () -> model.run(input,null,negative));
            assertThrows(IllegalArgumentException.class, () -> model.run(input,null,wrongShape));
            assertThrows(IllegalArgumentException.class, () -> model.run(input,wrongState));
            assertThrows(IllegalArgumentException.class, () -> model.run(integers));
            assertThrows(IllegalArgumentException.class, () -> model.run(foreign));
            assertThrows(IllegalArgumentException.class, () -> model.run(empty));
            assertThrows(IllegalArgumentException.class, () -> RecurrentState.fromSnapshot(context,Map.of("hidden",new TensorSnapshot(Shape.of(1,1),new int[]{1}))));
            assertEquals(before,context.liveAllocations());
            try (var output=model.forward(input)) { assertEquals(Shape.of(2,3,3),output.shape()); }
        }
    }

    @Test void resultAndStateOwnershipSurviveModelClosureAndCloseIdempotently() {
        try (var context=ExecutionContext.cpu()) {
            var model=definition(RecurrentCell.Kind.LSTM,SequenceOutput.ALL).build(context);
            var input=context.tensor(RecurrentCellTest.values(4),Shape.of(1,2,2)); var result=model.run(input); var state=result.state(); var output=result.output();
            result.close(); result.close(); model.close(); input.close();
            assertThrows(IllegalStateException.class,result::output); assertEquals(6,output.toFloatArray().length);
            try (var hidden=state.hidden()) { assertEquals(Shape.of(1,3),hidden.shape()); }
            output.close(); state.close(); state.close(); assertEquals(0,context.liveAllocations());
        }
    }

    private static void sameState(RecurrentState a,RecurrentState b) {
        try (var ah=a.hidden(); var bh=b.hidden(); var ac=a.hasCell() ? a.cell() : null; var bc=b.hasCell() ? b.cell() : null) {
            assertArrayEquals(ah.toFloatArray(),bh.toFloatArray()); if (ac!=null) { assertArrayEquals(ac.toFloatArray(),bc.toFloatArray()); }
        }
    }
}

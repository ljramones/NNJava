package org.dynamisengine.nn;

import java.util.LinkedHashMap;
import java.util.Map;
import org.dynamisengine.nn.nn.Recurrent;
import org.dynamisengine.nn.nn.RecurrentCell;
import org.dynamisengine.nn.nn.RecurrentState;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class RecurrentCellTest {
    static float[] values(int count) {
        float[] values=new float[count];
        for (int i=0;i<count;i++) { values[i]=(i%13-6)*.043f; }
        return values;
    }

    static Map<String,float[]> weights(RecurrentCell cell) {
        var weights=new LinkedHashMap<String,float[]>();
        cell.namedParameters().forEach((name,parameter) -> { try (var value=parameter.value()) { weights.put(name,value.toFloatArray()); } });
        return weights;
    }

    @Test void allCellKindsMatchIndependentBatchedEquationsWithAndWithoutBias() {
        for (var kind:RecurrentCell.Kind.values()) for (boolean bias:new boolean[]{true,false}) {
            try (var context=ExecutionContext.cpu(); var cell=RecurrentCell.builder().kind(kind).inputFeatures(2).hiddenSize(3).bias(bias).seed(42).build(context);
                 var x=context.tensor(values(4),Shape.of(2,2)); var h=context.tensor(values(6),Shape.of(2,3));
                 var c=kind==RecurrentCell.Kind.LSTM ? context.tensor(new float[]{.3f,-.2f,.1f,.4f,.15f,-.3f},h.shape()) : null;
                 var initial=c==null ? RecurrentState.hidden(h) : RecurrentState.lstm(h,c); var next=cell.step(x,initial); var output=next.hidden()) {
                var expected=reference(kind,x.toFloatArray(),h.toFloatArray(),c==null ? null : c.toFloatArray(),weights(cell),2,2,3);
                assertArrayEquals(expected[0],output.toFloatArray(),2e-6f,kind.name());
                if (c!=null) { try (var actual=next.cell()) { assertArrayEquals(expected[1],actual.toFloatArray(),2e-6f); } }
                assertArrayEquals(values(6),h.toFloatArray());
                assertEquals(bias ? 4 : 2,cell.namedParameters().size());
            }
        }
    }

    @Test void lstmGateOrderAndBothBiasVectorsAreExplicit() {
        try (var context=ExecutionContext.cpu(); var cell=RecurrentCell.builder().kind(RecurrentCell.Kind.LSTM).inputFeatures(1).hiddenSize(1).build(context)) {
            assign(context,cell,"input.weight",new float[4]); assign(context,cell,"recurrent.weight",new float[4]);
            assign(context,cell,"input.bias",new float[]{.4f,-.6f,.8f,1.2f}); assign(context,cell,"recurrent.bias",new float[]{.1f,.2f,-.3f,.4f});
            try (var x=context.tensor(new float[]{.7f},Shape.of(1,1)); var h=context.tensor(new float[]{.2f},Shape.of(1,1));
                 var c=context.tensor(new float[]{.6f},Shape.of(1,1)); var initial=RecurrentState.lstm(h,c); var next=cell.step(x,initial);
                 var actualH=next.hidden(); var actualC=next.cell()) {
                double expectedC=sigmoid(-.4)*.6+sigmoid(.5)*Math.tanh(.5);
                assertEquals(expectedC,actualC.toFloatArray()[0],2e-6); assertEquals(sigmoid(1.6)*Math.tanh(expectedC),actualH.toFloatArray()[0],2e-6);
            }
        }
    }

    @Test void gruUsesResetAfterProjectionIncludingTheRecurrentCandidateBias() {
        try (var context=ExecutionContext.cpu(); var cell=RecurrentCell.builder().kind(RecurrentCell.Kind.GRU).inputFeatures(1).hiddenSize(1).build(context)) {
            assign(context,cell,"input.weight",new float[3]); assign(context,cell,"recurrent.weight",new float[]{0,0,.7f});
            assign(context,cell,"input.bias",new float[]{.4f,-.3f,.2f}); assign(context,cell,"recurrent.bias",new float[]{.1f,.2f,.8f});
            try (var x=context.tensor(new float[]{.5f},Shape.of(1,1)); var h=context.tensor(new float[]{.6f},Shape.of(1,1));
                 var initial=RecurrentState.hidden(h); var next=cell.step(x,initial); var actual=next.hidden()) {
                double r=sigmoid(.5), z=sigmoid(-.1), n=Math.tanh(.2+r*(.7*.6+.8));
                assertEquals(n+z*(.6-n),actual.toFloatArray()[0],2e-6);
                double other=Math.tanh(.2+.7*(r*.6)+.8);
                assertTrue(Math.abs(actual.toFloatArray()[0]-(other+z*(.6-other)))>.01);
            }
        }
    }

    @Test void fullBpttInputInitialStatesAndEveryParameterPassFiniteDifferencesForAllKinds() {
        for (var kind:RecurrentCell.Kind.values()) {
            try (var context=ExecutionContext.cpu(); var model=Recurrent.builder().kind(kind).inputFeatures(1).hiddenSize(2).seed(19).build(context);
                 var x=context.tensor(new float[]{.1f,.25f},Shape.of(1,2,1),true);
                 var h=context.tensor(new float[]{.2f,-.1f},Shape.of(1,2),true);
                 var c=kind==RecurrentCell.Kind.LSTM ? context.tensor(new float[]{.15f,.3f},Shape.of(1,2),true) : null) {
                try (var tape=context.gradientTape(); var loss=loss(model,x,h,c)) { tape.backward(loss); }
                var variables=new LinkedHashMap<String,Tensor>(); variables.put("input",x.retain()); variables.put("initialHidden",h.retain());
                if (c!=null) { variables.put("initialCell",c.retain()); }
                model.namedParameters().forEach((name,parameter) -> variables.put(name,parameter.value()));
                try {
                    for (var entry:variables.entrySet()) {
                        var variable=entry.getValue(); float[] original=variable.toFloatArray(), gradient;
                        try (var g=variable.gradient()) { gradient=g.toFloatArray(); }
                        for (int i=0;i<original.length;i++) {
                            float value=original[i], delta=.002f;
                            original[i]=value+delta; assign(context,variable,original); double plus;
                            try (var loss=loss(model,x,h,c)) { plus=loss.item(); }
                            original[i]=value-delta; assign(context,variable,original); double minus;
                            try (var loss=loss(model,x,h,c)) { minus=loss.item(); }
                            original[i]=value; assign(context,variable,original);
                            double expected=(plus-minus)/(2*delta);
                            assertEquals(expected,gradient[i],.0003+Math.abs(expected)*.01,kind+" "+entry.getKey()+"["+i+"]");
                        }
                    }
                } finally { variables.values().forEach(Tensor::close); }
            }
        }
    }

    private static Tensor loss(Recurrent model,Tensor x,Tensor h,Tensor c) {
        try (var initial=c==null ? RecurrentState.hidden(h) : RecurrentState.lstm(h,c); var result=model.run(x,initial);
             var output=result.output(); var square=Ops.square(output); var sum=Ops.sum(square); var state=result.state();
             var hidden=state.hidden(); var hs=Ops.sum(hidden); var scaled=Ops.scale(hs,.25f); var total=Ops.add(sum,scaled)) {
            if (c==null) { return total.retain(); }
            try (var cell=state.cell(); var cs=Ops.sum(cell); var extra=Ops.scale(cs,.15f)) { return Ops.add(total,extra); }
        }
    }

    private static void assign(ExecutionContext context,RecurrentCell cell,String name,float[] values) {
        try (var value=cell.namedParameters().get(name).value()) { assign(context,value,values); }
    }
    private static void assign(ExecutionContext context,Tensor tensor,float[] values) {
        try (var source=context.tensor(values,tensor.shape())) { context.assign(tensor,source); }
    }

    static float[][] reference(RecurrentCell.Kind kind,float[] x,float[] hidden,float[] cell,Map<String,float[]> weights,int batch,int inputs,int width) {
        int gateWidth=kind.gates()*width;
        float[] a=project(x,weights.get("input.weight"),weights.get("input.bias"),batch,inputs,gateWidth);
        float[] b=project(hidden,weights.get("recurrent.weight"),weights.get("recurrent.bias"),batch,width,gateWidth);
        float[] h=new float[hidden.length], c=cell==null ? null : new float[cell.length];
        for (int row=0;row<batch;row++) for (int j=0;j<width;j++) {
            int offset=row*gateWidth+j, at=row*width+j;
            if (kind==RecurrentCell.Kind.RNN) { h[at]=(float)Math.tanh(a[offset]+b[offset]); }
            else if (kind==RecurrentCell.Kind.LSTM) {
                double i=sigmoid(a[offset]+b[offset]), f=sigmoid(a[offset+width]+b[offset+width]);
                double g=Math.tanh(a[offset+2*width]+b[offset+2*width]), o=sigmoid(a[offset+3*width]+b[offset+3*width]);
                c[at]=(float)(f*cell[at]+i*g); h[at]=(float)(o*Math.tanh(c[at]));
            } else {
                double r=sigmoid(a[offset]+b[offset]), z=sigmoid(a[offset+width]+b[offset+width]);
                double n=Math.tanh(a[offset+2*width]+r*b[offset+2*width]);
                h[at]=(float)(n+z*(hidden[at]-n));
            }
        }
        return new float[][]{h,c};
    }

    private static float[] project(float[] input,float[] weights,float[] bias,int batch,int inputs,int outputs) {
        float[] result=new float[batch*outputs];
        for (int row=0;row<batch;row++) for (int out=0;out<outputs;out++) {
            double sum=0;
            for (int i=0;i<inputs;i++) { sum+=(double)input[row*inputs+i]*weights[out*inputs+i]; }
            result[row*outputs+out]=(float)sum+(bias==null ? 0 : bias[out]);
        }
        return result;
    }
    private static double sigmoid(double x) { return 1/(1+Math.exp(-x)); }
}

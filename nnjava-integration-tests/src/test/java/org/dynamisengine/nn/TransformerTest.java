package org.dynamisengine.nn;

import static org.junit.jupiter.api.Assertions.*;
import java.util.*;
import org.dynamisengine.nn.transformer.*;
import org.dynamisengine.nn.nn.*;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.*;
import org.dynamisengine.nn.training.*;
import org.junit.jupiter.api.Test;

class TransformerTest {
    @Test void groupedCausalAttentionMatchesIndependentDoubleReferenceAndGradients() {
        float[] q={.1f,.3f,.4f,-.2f,.6f,-.1f,.3f,.7f};float[] k={.2f,-.5f,.6f,.4f};float[] v={.1f,.9f,-.3f,.7f};
        try(var c=ExecutionContext.cpu()) {
            try(var qt=c.tensor(q,Shape.of(1,2,2,2));var kt=c.tensor(k,Shape.of(1,1,2,2));var vt=c.tensor(v,Shape.of(1,1,2,2));
                    var output=Attention.scaledDotProduct(qt,kt,vt,null,true)) {
                float[] expected=new float[8];
                for(int head=0;head<2;head++) {for(int t=0;t<2;t++) {
                    double[] scores=new double[t+1];double total=0;
                    for(int s=0;s<=t;s++) {scores[s]=Math.exp((q[head*4+t*2]*k[s*2]+q[head*4+t*2+1]*k[s*2+1])/Math.sqrt(2));total+=scores[s];}
                    for(int d=0;d<2;d++) {double sum=0;for(int s=0;s<=t;s++) {sum+=scores[s]/total*v[s*2+d];}expected[head*4+t*2+d]=(float)sum;}
                }}
                assertArrayEquals(expected,output.toFloatArray(),1e-6f);
            }
            GradientChecks.check(c,new float[][] {q,k,v},new Shape[] {Shape.of(1,2,2,2),Shape.of(1,1,2,2),Shape.of(1,1,2,2)},args -> {
                try(var output=Attention.scaledDotProduct(args.get(0),args.get(1),args.get(2),null,true);var squared=Ops.square(output)) {return Ops.sum(squared);}
            });
            assertEquals(0,c.liveAllocations());
        }
    }

    @Test void fullAndCachedMultiHeadExecutionAgreeForBatchAndChunkedPrefill() {
        try(var c=ExecutionContext.cpu();var model=new MultiHeadAttention(c,4,2,1,33);
                var input=c.normal(Shape.of(2,4,4),12,.2f,false);var cache=new KvCache(c,2,1,4,2)) {
            model.training(false);
            try(var full=model.forward(input)) {
                List<Tensor> pieces=new ArrayList<>();
                try {
                    for(int start=0;start<4;start+=2) {try(var chunk=Ops.slice(input,1,start,start+2)) {pieces.add(model.decode(chunk,cache,null));}}
                    try(var joined=Ops.concat(1,pieces.toArray(Tensor[]::new))) {assertArrayEquals(full.toFloatArray(),joined.toFloatArray(),2e-5f);}
                } finally {pieces.forEach(Tensor::close);}
                assertEquals(4,cache.length());
                try(var extra=Ops.slice(input,1,0,1)) {assertThrows(IllegalArgumentException.class,()->model.decode(extra,cache,null));}
                cache.reset();assertEquals(0,cache.length());
                try(var tape=c.gradientTape()) {assertThrows(IllegalStateException.class,cache::reset);assertThrows(IllegalStateException.class,()->model.decode(input,cache,null));}
            }
        }
    }

    @Test void paddingMasksAndFullyMaskedQueriesHaveZeroOutputsAndGradients() {
        try(var c=ExecutionContext.cpu();var q=c.tensor(new float[] {1,2,3,4},Shape.of(1,1,2,2),true);
                var k=c.tensor(new float[] {1,0,0,1},Shape.of(1,1,2,2),true);var v=c.tensor(new float[] {2,4,6,8},Shape.of(1,1,2,2),true);
                var mask=c.tensor(new boolean[] {false,false,false,false},Shape.of(2,2));var tape=c.gradientTape();
                var y=Attention.scaledDotProduct(q,k,v,mask,true);var loss=Ops.sum(y)) {
            assertArrayEquals(new float[4],y.toFloatArray());tape.backward(loss);
            try(var dq=q.gradient();var dk=k.gradient();var dv=v.gradient()) {assertArrayEquals(new float[4],dq.toFloatArray());assertArrayEquals(new float[4],dk.toFloatArray());assertArrayEquals(new float[4],dv.toFloatArray());}
            try(var lengths=c.tensor(new int[] {1,2},Shape.of(2));var padding=Masks.padding(lengths,3)) {assertArrayEquals(new boolean[] {true,false,false,true,true,false},padding.toBooleanArray());}
        }
    }

    @Test void callerDefinedTinyTransformerTrainsUsingOnlyPublicPrimitives() {
        try(var c=ExecutionContext.cpu();var model=new TinyTransformer(c);var optimizer=new AdamW(model.namedParameters(),.025f);
                var tokens=c.tensor(new int[] {0,1,2,1,2,3},Shape.of(2,3));var labels=c.tensor(new int[] {1,2,3,2,3,4},Shape.of(2,3))) {
            float first=0,last=0;
            for(int i=0;i<60;i++) {last=CpuTrainingAndStateTest.step(c,model,optimizer,tokens,labels);if(i==0){first=last;}}
            assertTrue(last<first*.15f,"loss "+first+" -> "+last);
            optimizer.zeroGrad();model.training(false);
            try(var prediction=model.forward(tokens)) {assertEquals(Shape.of(2,3,5),prediction.shape());assertTrue(Metrics.accuracy(prediction,labels,-100)>.9);}
        }
    }

    private static final class TinyTransformer extends AbstractModule implements UnaryModule {
        final Embedding embedding;final RmsNorm norm1,norm2;final MultiHeadAttention attention;final GatedFeedForward feedForward;final Linear output;
        TinyTransformer(ExecutionContext c) {
            super(c);embedding=register("embedding",new Embedding(c,5,4,1));norm1=register("norm1",new RmsNorm(c,4,1e-5f));
            attention=register("attention",new MultiHeadAttention(c,4,2,1,2));norm2=register("norm2",new RmsNorm(c,4,1e-5f));
            feedForward=register("feedForward",new GatedFeedForward(c,4,8,3));output=register("output",new Linear(c,embedding.weight(),false));
        }
        public Tensor forward(Tensor tokens) {
            try(var x=embedding.forward(tokens);var n1=norm1.forward(x);var attended=attention.forward(n1);var residual=Ops.add(x,attended);
                    var n2=norm2.forward(residual);var feed=feedForward.forward(n2);var hidden=Ops.add(residual,feed)) {return output.forward(hidden);}
        }
    }
}

package org.dynamisengine.nn;

import static org.junit.jupiter.api.Assertions.*;
import java.util.List;
import java.util.function.Function;
import org.dynamisengine.nn.ops.*;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.*;
import org.junit.jupiter.api.Test;

class ExpandedCpuTest {
    @Test void splitViewsAndNegativeTransposeAxesPreserveGradients() {
        try(var c=ExecutionContext.cpu();var input=c.tensor(new float[] {1,2,3,4,5,6},Shape.of(2,3),true);var tape=c.gradientTape()) {
            List<Tensor> parts=Ops.split(input,-1,1,2);
            try(var joined=Ops.concat(-1,parts.toArray(Tensor[]::new));var transposed=Ops.transpose(joined,-2,-1);var total=Ops.sum(transposed)) {
                parts.forEach(Tensor::close);tape.backward(total);
                try(var gradient=input.gradient()) {assertArrayEquals(new float[] {1,1,1,1,1,1},gradient.toFloatArray());}
            }
        }
    }
    @Test void batchedGemmBroadcastsAndReducesParameterGradients() {
        try (var c = ExecutionContext.cpu()) {
            float[] av = {1,2,3,4,5,6,7,8,9,10,11,12};
            float[] bv = {1,2,3,4,5,6};
            try (var a = c.tensor(av, Shape.of(2,2,3)); var b = c.tensor(bv, Shape.of(3,2)); var y = Ops.matmul(a,b)) {
                assertArrayEquals(new float[] {22,28,49,64,76,100,103,136}, y.toFloatArray());
            }
            GradientChecks.check(c, new float[][] {{.2f,.4f,.1f,.3f}, {.6f,.7f}},
                    new Shape[] {Shape.of(2,1,2), Shape.of(2,1)}, v -> {
                        try (var y = Ops.matmul(v.get(0),v.get(1)); var square = Ops.square(y)) { return Ops.sum(square); }
                    });
            assertEquals(0,c.liveAllocations());
        }
    }

    @Test void axisReductionsKeepDimensionsAndChooseFirstMaximum() {
        try (var c = ExecutionContext.cpu(); var x = c.tensor(new float[] {1,4,4,3,2,0}, Shape.of(2,3));
                var sum = Ops.sum(x,0,true); var mean = Ops.mean(x,1,false); var max = Ops.max(x,1,false); var arg = Ops.argmax(x,1,false)) {
            assertEquals(Shape.of(1,3),sum.shape());
            assertArrayEquals(new float[] {4,6,4},sum.toFloatArray());
            assertArrayEquals(new float[] {3,5f/3},mean.toFloatArray(),1e-6f);
            assertArrayEquals(new float[] {4,3},max.toFloatArray());
            assertArrayEquals(new int[] {1,0},arg.toIntArray());
        }
        try (var c = ExecutionContext.cpu()) {
            GradientChecks.check(c,new float[][] {{.1f,.8f,-.3f,.7f,-.2f,.4f}},new Shape[] {Shape.of(2,3)},v -> {
                try(var max=Ops.max(v.getFirst(),1,true);var mean=Ops.mean(v.getFirst(),0,false);var a=Ops.sum(max);var b=Ops.sum(mean)) { return Ops.add(a,b); }
            });
            try(var x=c.tensor(new float[] {1,Float.NaN,Float.NaN},Shape.of(3));var maximum=Ops.max(x,0,false);var index=Ops.argmax(x,0,false)) {
                assertTrue(Float.isNaN(maximum.item()));assertArrayEquals(new int[] {1},index.toIntArray());
            }
        }
    }

    @Test void indexingConcatenationAndSlicesHaveAccurateBackwardRules() {
        try(var c=ExecutionContext.cpu();var indices=c.tensor(new int[] {2,0,2},Shape.of(3))) {
            GradientChecks.check(c,new float[][] {{.1f,.2f,.3f,.4f,.5f,.6f}},new Shape[] {Shape.of(3,2)},v -> {
                try(var gathered=Ops.gather(v.getFirst(),0,indices);var slice=Ops.slice(v.getFirst(),0,0,3,2);
                        var cat=Ops.concat(0,gathered,slice);var transposed=Ops.permute(cat,1,0);var square=Ops.square(transposed)) { return Ops.sum(square); }
            });
            assertEquals(1,c.liveAllocations());
        }
    }

    @Test void masksCastsAndSelectionPreserveTypedValuesAndGradients() {
        try(var c=ExecutionContext.cpu();var mask=c.tensor(new boolean[] {true,false,true},Shape.of(3))) {
            GradientChecks.check(c,new float[][] {{.2f,.4f,.6f},{.1f}},new Shape[] {Shape.of(3),Shape.of()},v -> {
                try(var selected=Ops.where(mask,v.get(0),v.get(1));var square=Ops.square(selected)) { return Ops.sum(square); }
            });
            try(var ints=c.tensor(new int[] {1,2,3},Shape.of(3));var floats=Ops.cast(ints,DType.FP32);
                    var comparison=Ops.equal(ints,ints);var inverse=Ops.not(mask);var combined=Ops.and(comparison,inverse)) {
                assertArrayEquals(new float[] {1,2,3},floats.toFloatArray());
                assertArrayEquals(new boolean[] {false,true,false},combined.toBooleanArray());
            }
        }
    }

    @Test void vectrixActivationsAndElementaryFunctionsHaveAccurateDerivatives() {
        List<Function<Tensor,Tensor>> functions=List.of(Ops::relu,Ops::sigmoid,Ops::tanh,Ops::gelu,Ops::silu,Ops::exp,Ops::log,Ops::sqrt);
        try(var c=ExecutionContext.cpu()) {
            for(var function:functions) {
                GradientChecks.check(c,new float[][] {{.2f,.6f,1.3f}},new Shape[] {Shape.of(3)},v -> {
                    try(var y=function.apply(v.getFirst());var square=Ops.square(y)) { return Ops.sum(square); }
                });
            }
            GradientChecks.check(c,new float[][] {{.2f,.7f},{.3f}},new Shape[] {Shape.of(2),Shape.of()},v -> {
                try(var y=Ops.divide(v.get(0),v.get(1));var square=Ops.square(y)) { return Ops.sum(square); }
            });
            assertEquals(0,c.liveAllocations());
        }
    }

    @Test void softmaxWorksAcrossAxesAndFullyMaskedRowsAreZero() {
        try(var c=ExecutionContext.cpu()) {
            GradientChecks.check(c,new float[][] {{.1f,.4f,.7f,.2f}},new Shape[] {Shape.of(2,2)},v -> {
                try(var p=Ops.softmax(v.getFirst(),0);var q=Ops.logSoftmax(v.getFirst(),-1);var a=Ops.square(p);var b=Ops.sum(a);var d=Ops.sum(q)) {return Ops.add(b,d);}
            });
            try(var x=c.tensor(new float[] {Float.NEGATIVE_INFINITY,Float.NEGATIVE_INFINITY,0,0},Shape.of(2,2),true);
                    var tape=c.gradientTape();var p=Ops.softmax(x,-1);var total=Ops.sum(p)) {
                assertArrayEquals(new float[] {0,0,.5f,.5f},p.toFloatArray());
                tape.backward(total);
                try(var dx=x.gradient()) {assertArrayEquals(new float[4],dx.toFloatArray(),1e-7f);}
            }
        }
    }

    @Test void normalizationDifferentiatesInputScaleAndBias() {
        try(var c=ExecutionContext.cpu()) {
            float[][] values={{.2f,-.7f,.4f,.8f,.1f,-.3f},{.8f,1.1f,.9f},{.1f,-.2f,.3f}};
            Shape[] shapes={Shape.of(2,3),Shape.of(3),Shape.of(3)};
            GradientChecks.check(c,values,shapes,v -> {
                try(var y=Ops.layerNorm(v.get(0),v.get(1),v.get(2),1e-5f);var z=Ops.square(y)) {return Ops.sum(z);}
            });
            GradientChecks.check(c,new float[][] {values[0],values[1]},new Shape[] {shapes[0],shapes[1]},v -> {
                try(var y=Ops.rmsNorm(v.get(0),v.get(1),1e-5f);var z=Ops.square(y)) {return Ops.sum(z);}
            });
            assertEquals(0,c.liveAllocations());
        }
    }

    @Test void stableCrossEntropySupportsIgnoredAndUnreducedTargets() {
        try(var c=ExecutionContext.cpu();var labels=c.tensor(new int[] {0,-100},Shape.of(2))) {
            GradientChecks.check(c,new float[][] {{1.1f,-.4f,.3f,.2f}},new Shape[] {Shape.of(2,2)},
                    v -> NeuralOps.crossEntropy(v.getFirst(),labels,"mean",-100));
            try(var logits=c.tensor(new float[] {1000,1000,0,0},Shape.of(2,2));var loss=NeuralOps.crossEntropy(logits,labels,"none",-100)) {
                assertArrayEquals(new float[] {(float)Math.log(2),0},loss.toFloatArray(),1e-4f);
            }
            try(var ignored=c.tensor(new int[] {-100,-100},Shape.of(2));var logits=c.full(Shape.of(2,2),1);
                    var loss=NeuralOps.crossEntropy(logits,ignored,"mean",-100)) {assertEquals(0,loss.item());}
        }
    }

    @Test void dropoutReplaysItsMaskForBackwardAndRotaryGradientsMatch() {
        try(var c=ExecutionContext.cpu();var x=c.tensor(new float[] {1,1,1,1,1,1,1,1},Shape.of(8),true);
                var tape=c.gradientTape();var y=Ops.dropout(x,.5f,123,0);var total=Ops.sum(y)) {
            tape.backward(total);
            try(var dx=x.gradient()) {assertArrayEquals(y.toFloatArray(),dx.toFloatArray());}
        }
        try(var c=ExecutionContext.cpu();var positions=c.tensor(new int[] {4,7},Shape.of(2))) {
            GradientChecks.check(c,new float[][] {{.2f,-.1f,.7f,.4f,.6f,-.2f,.3f,.8f}},new Shape[] {Shape.of(1,2,4)},v -> {
                try(var y=Ops.rope(v.getFirst(),positions,10000,true);var e=Ops.exp(y)) {return Ops.sum(e);}
            });
        }
    }
}

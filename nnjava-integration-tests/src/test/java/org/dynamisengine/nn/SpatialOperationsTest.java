package org.dynamisengine.nn;

import java.util.Arrays;
import java.util.List;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.ops.Conv2dConfig;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.ops.Pool2dConfig;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class SpatialOperationsTest {
    private static float[] values(int size) {
        float[] result=new float[size];
        for (int i=0;i<size;i++) { result[i]=(i%17-8)*.037f; }
        return result;
    }

    @Test void convolutionMatchesAnIndependentReferenceForBatchesGroupsDepthwiseAndRectangularGeometry() {
        for (int groups : new int[]{1,2,4}) {
            var config=new Conv2dConfig(2,2).stride(2,1).padding(1,0).dilation(1,2).groups(groups);
            Shape inputShape=Shape.of(2,4,4,6), weightShape=config.weightShape(4,8);
            float[] input=values((int)inputShape.elements()), weights=values((int)weightShape.elements()), bias=values(8);
            try (var context=ExecutionContext.cpu(); var x=context.tensor(input,inputShape); var w=context.tensor(weights,weightShape);
                 var b=context.tensor(bias,Shape.of(8)); var output=Ops.conv2d(x,w,b,config)) {
                assertEquals(Shape.of(2,8,3,4),output.shape());
                assertArrayEquals(convolve(input,weights,bias,inputShape,8,config),output.toFloatArray(),2e-6f);
            }
        }
    }

    @Test void adapterMatchesDirectVectrixForwardAndFixedOrderBatchedWeightAndBiasGradients() {
        var config=new Conv2dConfig(2,3).padding(1,1).groups(2);
        Shape inputShape=Shape.of(2,4,3,4), weightShape=config.weightShape(4,4);
        float[] input=values((int)inputShape.elements()), weights=values((int)weightShape.elements());
        try (var context=ExecutionContext.cpu(); var x=context.tensor(input,inputShape,true); var w=context.tensor(weights,weightShape,true);
             var b=context.tensor(new float[4],Shape.of(4),true)) {
            float[] actual;
            Shape outShape;
            try (var tape=context.gradientTape(); var output=Ops.conv2d(x,w,b,config); var loss=Ops.sum(output)) {
                actual=output.toFloatArray(); outShape=output.shape(); tape.backward(loss);
            }
            var plan=org.dynamisengine.vectrix.nn.Conv2d.of(4,4,3,4,2,3,1,1,1,2);
            int inputSize=48, outputSize=(int)(outShape.elements()/2);
            float[] expected=new float[actual.length], dx=new float[input.length], dw=new float[weights.length], db=new float[4];
            float[] dy=new float[actual.length], scratch=new float[plan.scratchSize()]; Arrays.fill(dy,1);
            for (int i=0;i<2;i++) {
                plan.forward(input,i*inputSize,weights,0,new float[4],0,expected,i*outputSize,scratch,0);
                plan.backward(input,i*inputSize,weights,0,dy,i*outputSize,dx,i*inputSize,dw,0,db,0,scratch,0);
            }
            assertArrayEquals(expected,actual);
            try (var gx=x.gradient(); var gw=w.gradient(); var gb=b.gradient()) {
                assertArrayEquals(dx,gx.toFloatArray()); assertArrayEquals(dw,gw.toFloatArray()); assertArrayEquals(db,gb.toFloatArray());
            }
        }
    }

    @Test void groupedConvolutionInputWeightAndBiasGradientsPassFiniteDifferences() {
        var config=new Conv2dConfig(2,2).stride(2,1).padding(1).groups(2);
        Shape[] shapes={Shape.of(2,2,3,4),config.weightShape(2,2),Shape.of(2)};
        try (var context=ExecutionContext.cpu()) {
            GradientChecks.check(context,new float[][]{values(48),values(8),values(2)},shapes,tensors -> {
                try (var output=Ops.conv2d(tensors.get(0),tensors.get(1),tensors.get(2),config); var square=Ops.square(output)) { return Ops.mean(square); }
            });
            assertEquals(0,context.liveAllocations());
        }
    }

    @Test void noncontiguousInputsWeightsAndBiasesPreserveForwardAndBackwardResults() {
        var config=new Conv2dConfig(2,2).padding(1);
        try (var context=ExecutionContext.cpu(); var base=context.tensor(values(48),Shape.of(2,2,4,3),true);
             var x=Ops.transpose(base,2,3); var weightBase=context.tensor(values(32),Shape.of(4,2,2,2),true);
             var selected=Ops.slice(weightBase,0,1,3,1); var weight=Ops.transpose(selected,2,3);
             var biasBase=context.tensor(values(4),Shape.of(4),true); var bias=Ops.slice(biasBase,0,1,3,1);
             var directX=context.tensor(x.toFloatArray(),x.shape(),true); var directW=context.tensor(weight.toFloatArray(),weight.shape(),true);
             var directB=context.tensor(bias.toFloatArray(),bias.shape(),true)) {
            try (var tape=context.gradientTape(); var trackedX=Ops.transpose(base,2,3); var trackedSelected=Ops.slice(weightBase,0,1,3,1);
                 var trackedWeight=Ops.transpose(trackedSelected,2,3); var trackedBias=Ops.slice(biasBase,0,1,3,1);
                 var actual=Ops.conv2d(trackedX,trackedWeight,trackedBias,config); var loss=Ops.sum(actual)) {
                try (var expected=Ops.conv2d(directX,directW,directB,config)) { assertArrayEquals(expected.toFloatArray(),actual.toFloatArray()); }
                tape.backward(loss);
            }
            try (var tape=context.gradientTape(); var expected=Ops.conv2d(directX,directW,directB,config); var loss=Ops.sum(expected)) { tape.backward(loss); }
            // View gradients are accumulated into the trainable bases through their own backward rules.
            try (var gx=base.gradient(); var transposed=Ops.transpose(gx,2,3); var expected=directX.gradient()) {
                assertArrayEquals(expected.toFloatArray(),transposed.toFloatArray());
            }
            try (var gw=weightBase.gradient(); var slice=Ops.slice(gw,0,1,3,1); var transposed=Ops.transpose(slice,2,3); var expected=directW.gradient()) {
                assertArrayEquals(expected.toFloatArray(),transposed.toFloatArray());
            }
        }
    }

    @Test void convolutionSupportsUnbatchedAndMultipleBatchAxesWithoutBias() {
        var config=new Conv2dConfig(1,1);
        for (Shape shape : List.of(Shape.of(2,3,4),Shape.of(2,3,2,3,4))) {
            try (var context=ExecutionContext.cpu(); var x=context.tensor(values((int)shape.elements()),shape);
                 var w=context.tensor(values(6),Shape.of(3,2,1,1)); var output=Ops.conv2d(x,w,config)) {
                assertEquals(shape.prefix(shape.rank()-3).append(3,3,4),output.shape());
                assertArrayEquals(convolve(x.toFloatArray(),w.toFloatArray(),null,shape,3,config),output.toFloatArray(),1e-6f);
            }
        }
    }

    @Test void onlyWatchedOperandsReceiveGradientsAndSharedUsesAccumulate() {
        try (var context=ExecutionContext.cpu(); var x=context.tensor(new float[]{1,2,3,4},Shape.of(1,2,2));
             var w=context.tensor(new float[]{2},Shape.of(1,1,1,1),true); var tape=context.gradientTape();
             var y=Ops.conv2d(x,w,new Conv2dConfig(1,1)); var twice=Ops.add(y,y); var loss=Ops.sum(twice)) {
            tape.backward(loss); assertFalse(x.hasGradient());
            try (var gradient=w.gradient()) { assertArrayEquals(new float[]{20},gradient.toFloatArray()); }
        }
        try (var context=ExecutionContext.cpu(); var x=context.tensor(new float[]{1,2,3,4},Shape.of(1,2,2),true);
             var w=context.tensor(new float[]{2},Shape.of(1,1,1,1)); var tape=context.gradientTape();
             var y=Ops.conv2d(x,w,new Conv2dConfig(1,1)); var loss=Ops.sum(y)) {
            tape.backward(loss); assertFalse(w.hasGradient());
            try (var gradient=x.gradient()) { assertArrayEquals(new float[]{2,2,2,2},gradient.toFloatArray()); }
        }
    }

    @Test void emptyBatchesHaveEmptyOutputsAndZeroParameterGradients() {
        var config=new Conv2dConfig(2,2);
        try (var context=ExecutionContext.cpu(); var x=context.tensor(new float[0],Shape.of(0,2,3,3),true);
             var w=context.tensor(values(24),Shape.of(3,2,2,2),true); var b=context.tensor(values(3),Shape.of(3),true)) {
            try (var tape=context.gradientTape(); var output=Ops.conv2d(x,w,b,config); var loss=Ops.sum(output)) {
                assertEquals(Shape.of(0,3,2,2),output.shape()); tape.backward(loss);
            }
            try (var gw=w.gradient(); var gb=b.gradient(); var gx=x.gradient(); var pool=Ops.maxPool2d(x,new Pool2dConfig(2))) {
                assertArrayEquals(new float[24],gw.toFloatArray()); assertArrayEquals(new float[3],gb.toFloatArray());
                assertEquals(0,gx.shape().elements()); assertEquals(0,pool.shape().elements());
            }
        }
    }

    @Test void maxAndAveragePoolingMatchIndependentPaddedOverlappingWindowsAndGradients() {
        Shape shape=Shape.of(2,2,3,4); float[] values=values(48);
        for (boolean average : new boolean[]{false,true}) {
            var config=new Pool2dConfig(3,1,1);
            var reference=pool(values,shape,config,average);
            try (var context=ExecutionContext.cpu(); var x=context.tensor(values,shape,true)) {
                try (var tape=context.gradientTape(); var output=average ? Ops.avgPool2d(x,config) : Ops.maxPool2d(x,config); var loss=Ops.sum(output)) {
                    assertArrayEquals(reference[0],output.toFloatArray(),1e-6f); tape.backward(loss);
                }
                try (var gradient=x.gradient()) { assertArrayEquals(reference[1],gradient.toFloatArray(),1e-6f); }
            }
        }
    }

    @Test void averagePoolingGradientPassesFiniteDifferencesWithPadding() {
        try (var context=ExecutionContext.cpu()) {
            GradientChecks.check(context,new float[][]{values(12)},new Shape[]{Shape.of(1,3,4)},tensors -> {
                try (var output=Ops.avgPool2d(tensors.getFirst(),new Pool2dConfig(3,1,1)); var square=Ops.square(output)) { return Ops.mean(square); }
            });
            assertEquals(0,context.liveAllocations());
        }
    }

    @Test void maxPoolingFirstTieFirstNanAndNegativePaddingSemanticsArePreserved() {
        for (float[] values : new float[][]{{4,4,3,2},{Float.NaN,Float.NaN,3,2},{Float.NEGATIVE_INFINITY,Float.NEGATIVE_INFINITY,Float.NEGATIVE_INFINITY,Float.NEGATIVE_INFINITY}}) {
            try (var context=ExecutionContext.cpu(); var x=context.tensor(values,Shape.of(1,2,2),true)) {
                try (var tape=context.gradientTape(); var output=Ops.maxPool2d(x,new Pool2dConfig(2)); var loss=Ops.sum(output)) {
                    if (Float.isNaN(values[0])) { assertTrue(Float.isNaN(output.toFloatArray()[0])); } else { assertEquals(values[0],output.toFloatArray()[0]); }
                    tape.backward(loss);
                }
                try (var gradient=x.gradient()) { assertArrayEquals(new float[]{1,0,0,0},gradient.toFloatArray()); }
            }
        }
        try (var context=ExecutionContext.cpu(); var x=context.tensor(new float[]{-5},Shape.of(1,1,1));
             var max=Ops.maxPool2d(x,new Pool2dConfig(2,1,1)); var avg=Ops.avgPool2d(x,new Pool2dConfig(2,1,1))) {
            assertArrayEquals(new float[]{-5,-5,-5,-5},max.toFloatArray());
            assertArrayEquals(new float[]{-5,-5,-5,-5},avg.toFloatArray());
        }
    }

    @Test void maxPoolingGradientPassesFiniteDifferencesAwayFromTies() {
        float[] input=new float[12]; for (int i=0;i<input.length;i++) { input[i]=i*.1f-.4f; }
        try (var context=ExecutionContext.cpu()) {
            GradientChecks.check(context,new float[][]{input},new Shape[]{Shape.of(1,3,4)},tensors -> {
                try (var output=Ops.maxPool2d(tensors.getFirst(),new Pool2dConfig(2,1,0)); var square=Ops.square(output)) { return Ops.mean(square); }
            });
        }
    }

    @Test void poolingPacksStridedViewsAndRoutesGradientsToTheOriginalStorage() {
        for (boolean average : new boolean[]{false,true}) {
            var config=new Pool2dConfig(3,1,1);
            try (var context=ExecutionContext.cpu(); var base=context.tensor(values(24),Shape.of(2,4,3),true)) {
                float[][] expected;
                try (var view=Ops.transpose(base,1,2)) { expected=pool(view.toFloatArray(),view.shape(),config,average); }
                try (var tape=context.gradientTape(); var view=Ops.transpose(base,1,2);
                     var output=average ? Ops.avgPool2d(view,config) : Ops.maxPool2d(view,config); var loss=Ops.sum(output)) {
                    assertArrayEquals(expected[0],output.toFloatArray(),1e-6f); tape.backward(loss);
                }
                try (var gradient=base.gradient(); var transposed=Ops.transpose(gradient,1,2)) { assertArrayEquals(expected[1],transposed.toFloatArray(),1e-6f); }
            }
        }
    }

    @Test void poolingBackwardSupportsEmptyBatches() {
        for (boolean average : new boolean[]{false,true}) {
            try (var context=ExecutionContext.cpu(); var input=context.tensor(new float[0],Shape.of(0,2,3,4),true)) {
                try (var tape=context.gradientTape(); var output=average ? Ops.avgPool2d(input,new Pool2dConfig(2)) : Ops.maxPool2d(input,new Pool2dConfig(2)); var loss=Ops.sum(output)) {
                    tape.backward(loss);
                }
                try (var gradient=input.gradient()) { assertEquals(input.shape(),gradient.shape()); assertEquals(0,gradient.toFloatArray().length); }
            }
        }
    }

    @Test void spatialContractsRejectBadGeometryTypesSeedsAndGradientIndicesWithoutPoisoningTheContext() {
        assertThrows(IllegalArgumentException.class, () -> new Conv2dConfig(4,4).stride(2).outputShape(Shape.of(1,3,3),1));
        assertThrows(IllegalArgumentException.class, () -> new Pool2dConfig(4,2,0).outputShape(Shape.of(1,3,3)));
        assertThrows(IllegalArgumentException.class, () -> new Pool2dConfig(2,1,2));
        assertThrows(IllegalArgumentException.class, () -> new Conv2dConfig(2,2).groups(2).weightShape(3,4));
        assertThrows(ArithmeticException.class, () -> new Conv2dConfig(1,1).padding(1).outputShape(Shape.of(1,1,Long.MAX_VALUE),1));
        try (var context=ExecutionContext.cpu(); var x=context.tensor(values(9),Shape.of(1,3,3)); var w=context.tensor(values(4),Shape.of(1,1,2,2));
             var badSeed=context.full(Shape.of(1),1); var output=context.empty(DType.FP32,x.shape()); var integers=context.tensor(new int[9],x.shape())) {
            int count=context.liveAllocations(); var attrs=new Conv2dConfig(2,2).attributes().and("bias",false);
            assertThrows(IllegalArgumentException.class, () -> context.submit(Operation.CONV2D_BACKWARD,List.of(x,w,badSeed),output,0,attrs.and("which",0)));
            assertThrows(IllegalArgumentException.class, () -> context.submit(Operation.CONV2D_BACKWARD,List.of(x,w,badSeed),output,0,attrs.and("which",2)));
            assertThrows(IllegalArgumentException.class, () -> context.submit(Operation.MAX_POOL2D_BACKWARD,List.of(x,badSeed),output,0,new Pool2dConfig(2).attributes()));
            assertThrows(IllegalArgumentException.class, () -> Ops.avgPool2d(integers,new Pool2dConfig(2)));
            assertEquals(count,context.liveAllocations());
            try (var usable=context.full(Shape.of(),7)) { assertEquals(7,usable.item()); }
        }
    }

    @Test void upstreamScratchLimitsAreRejectedBeforeExecutionAndWithoutAllocatingHugeArrays() {
        try (var context=ExecutionContext.cpu(); var x=context.tensor(new float[0],Shape.of(0,1,8192,8192));
             var w=context.tensor(values(49),Shape.of(1,1,7,7))) {
            int before=context.liveAllocations();
            assertThrows(IllegalArgumentException.class, () -> Ops.conv2d(x,w,new Conv2dConfig(7,7)));
            assertEquals(before,context.liveAllocations());
            try (var usable=context.full(Shape.of(),3)) { assertEquals(3,usable.item()); }
        }
    }

    @Test void savedSpatialInputsCannotBeMutatedBeforeBackward() {
        try (var context=ExecutionContext.cpu(); var x=context.tensor(values(9),Shape.of(1,3,3),true); var w=context.tensor(values(4),Shape.of(1,1,2,2),true);
             var tape=context.gradientTape(); var convolution=Ops.conv2d(x,w,new Conv2dConfig(2,2));
             var pooled=Ops.maxPool2d(convolution,new Pool2dConfig(2)); var loss=Ops.sum(pooled)) {
            assertThrows(IllegalStateException.class, () -> context.fill(x,0)); tape.backward(loss);
        }
    }

    private static float[] convolve(float[] input,float[] weights,float[] bias,Shape shape,int outputs,Conv2dConfig c) {
        Shape image=Conv2dConfig.imageShape(shape), result=c.outputShape(shape,outputs);
        int channels=(int)image.dimension(0), h=(int)image.dimension(1), w=(int)image.dimension(2);
        int oh=(int)result.dimension(result.rank()-2), ow=(int)result.dimension(result.rank()-1), images=(int)shape.prefix(shape.rank()-3).elements();
        float[] output=new float[(int)result.elements()];
        for (int n=0;n<images;n++) for (int out=0;out<outputs;out++) for (int y=0;y<oh;y++) for (int x=0;x<ow;x++) {
            double sum=0; int group=out/(outputs/c.groups());
            for (int channel=0;channel<channels/c.groups();channel++) for (int ky=0;ky<c.kernelHeight();ky++) for (int kx=0;kx<c.kernelWidth();kx++) {
                int iy=y*c.strideHeight()-c.paddingHeight()+ky*c.dilationHeight(), ix=x*c.strideWidth()-c.paddingWidth()+kx*c.dilationWidth();
                if (iy>=0 && iy<h && ix>=0 && ix<w) {
                    int index=((n*channels+group*(channels/c.groups())+channel)*h+iy)*w+ix;
                    int wi=((out*(channels/c.groups())+channel)*c.kernelHeight()+ky)*c.kernelWidth()+kx;
                    sum+=(double)input[index]*weights[wi];
                }
            }
            output[((n*outputs+out)*oh+y)*ow+x]=(float)sum+(bias==null ? 0 : bias[out]);
        }
        return output;
    }

    private static float[][] pool(float[] input,Shape shape,Pool2dConfig c,boolean average) {
        Shape image=Conv2dConfig.imageShape(shape), result=c.outputShape(shape);
        int channels=(int)image.dimension(0), h=(int)image.dimension(1), w=(int)image.dimension(2);
        int oh=(int)result.dimension(result.rank()-2), ow=(int)result.dimension(result.rank()-1), images=(int)shape.prefix(shape.rank()-3).elements();
        float[] output=new float[(int)result.elements()], gradient=new float[input.length];
        for (int n=0;n<images;n++) for (int ch=0;ch<channels;ch++) for (int y=0;y<oh;y++) for (int x=0;x<ow;x++) {
            var indices=new java.util.ArrayList<Integer>();
            for (int ky=0;ky<c.window();ky++) for (int kx=0;kx<c.window();kx++) {
                int iy=y*c.stride()-c.padding()+ky, ix=x*c.stride()-c.padding()+kx;
                if (iy>=0 && iy<h && ix>=0 && ix<w) { indices.add(((n*channels+ch)*h+iy)*w+ix); }
            }
            float value=0;
            if (average) {
                for (int i : indices) { value+=input[i]; gradient[i]+=1f/indices.size(); }
                value/=indices.size();
            } else {
                int selected=indices.getFirst();
                for (int i : indices) { if (input[i]>input[selected] || Float.isNaN(input[i]) && !Float.isNaN(input[selected])) { selected=i; } }
                value=input[selected]; gradient[selected]++;
            }
            output[((n*channels+ch)*oh+y)*ow+x]=value;
        }
        return new float[][]{output,gradient};
    }
}

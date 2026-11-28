package org.dynamisengine.nn.backend.vectrix;

import java.util.Arrays;
import org.dynamisengine.nn.backend.KernelCall;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.ops.Conv2dConfig;
import org.dynamisengine.nn.ops.Pool2dConfig;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.vectrix.nn.Conv2d;
import org.dynamisengine.vectrix.nn.Pool2d;

/** Packs layouts and batches calls to Vectrix's spatial forward/backward kernels. */
final class CpuSpatialKernels {
    private CpuSpatialKernels() { }

    static boolean handles(Operation operation) {
        return switch (operation) {
            case CONV2D, CONV2D_BACKWARD, MAX_POOL2D, MAX_POOL2D_BACKWARD, AVG_POOL2D, AVG_POOL2D_BACKWARD -> true;
            default -> false;
        };
    }

    /** Check upstream plan/array limits before the runtime accepts the operation. */
    static void validate(KernelCall call) {
        if (call.operation()==Operation.CONV2D || call.operation()==Operation.CONV2D_BACKWARD) { convolutionPlan(call); }
        else { poolingPlan(call); }
    }

    private static Conv2d convolutionPlan(KernelCall call) {
        var config = Conv2dConfig.fromAttributes(call.attributes());
        Shape input = Conv2dConfig.imageShape(call.inputs().getFirst().layout().shape());
        Shape weights = call.inputs().get(1).layout().shape();
        Shape output = config.outputShape(input,weights.dimension(0));
        Math.toIntExact(input.elements()); Math.toIntExact(weights.elements()); Math.toIntExact(output.elements());
        long columns = Math.multiplyExact(weights.elements()/weights.dimension(0),Math.multiplyExact(output.dimension(1),output.dimension(2)));
        if (columns>Integer.MAX_VALUE) { throw new IllegalArgumentException("Vectrix convolution scratch exceeds the CPU array limit"); }
        return Conv2d.of(Math.toIntExact(input.dimension(0)),Math.toIntExact(weights.dimension(0)),Math.toIntExact(input.dimension(1)),Math.toIntExact(input.dimension(2)),
                config.kernelHeight(),config.kernelWidth(),config.strideHeight(),config.strideWidth(),config.paddingHeight(),config.paddingWidth(),config.dilationHeight(),config.dilationWidth(),config.groups());
    }

    private static Pool2d poolingPlan(KernelCall call) {
        var config = Pool2dConfig.fromAttributes(call.attributes());
        Shape input = Conv2dConfig.imageShape(call.inputs().getFirst().layout().shape());
        Math.toIntExact(input.elements()); Math.toIntExact(config.outputShape(input).elements());
        return Pool2d.of(Math.toIntExact(input.dimension(0)),Math.toIntExact(input.dimension(1)),Math.toIntExact(input.dimension(2)),config.window(),config.stride(),config.padding());
    }

    static void execute(KernelCall call, Object[] data, float[] output) {
        if (call.operation()==Operation.CONV2D || call.operation()==Operation.CONV2D_BACKWARD) { convolution(call,data,output); }
        else { pooling(call,data,output); }
    }

    private static void convolution(KernelCall call, Object[] data, float[] output) {
        var plan = convolutionPlan(call);
        Shape shape = call.inputs().getFirst().layout().shape();
        int imageSize = Math.toIntExact(Conv2dConfig.imageShape(shape).elements());
        int images = Math.toIntExact(shape.prefix(shape.rank()-3).elements());
        int outputSize = Math.toIntExact(call.inputs().get(1).layout().shape().dimension(0)*(long)plan.outHeight()*plan.outWidth());
        int offset = Math.toIntExact(call.output().layout().offset());
        boolean backward = call.operation()==Operation.CONV2D_BACKWARD;
        int which = backward ? call.attributes().intValue("which") : -1;
        if (backward) { Arrays.fill(output,offset,offset+Math.toIntExact(call.output().layout().shape().elements()),0); }
        if (images==0) { return; }
        float[] input = CpuTensorKernels.floats(call.inputs().getFirst(),data[0]);
        float[] weights = CpuTensorKernels.floats(call.inputs().get(1),data[1]);
        float[] bias = !backward && call.attributes().flag("bias") ? CpuTensorKernels.floats(call.inputs().get(2),data[2]) : null;
        float[] dy = backward ? CpuTensorKernels.floats(call.inputs().getLast(),data[data.length-1]) : null;
        float[] scratch = new float[plan.scratchSize()];
        for (int image=0;image<images;image++) {
            if (!backward) { plan.forward(input,image*imageSize,weights,0,bias,0,output,offset+image*outputSize,scratch,0); }
            else {
                // Vectrix accumulates dWeights/dBias in image order and writes each dInput independently.
                plan.backward(input,image*imageSize,weights,0,dy,image*outputSize,which==0 ? output : null,offset+image*imageSize,
                        which==1 ? output : null,offset,which==2 ? output : null,offset,scratch,0);
            }
        }
    }

    private static void pooling(KernelCall call, Object[] data, float[] output) {
        var plan = poolingPlan(call);
        Shape shape = call.inputs().getFirst().layout().shape();
        int imageSize = Math.toIntExact(Conv2dConfig.imageShape(shape).elements());
        int images = Math.toIntExact(shape.prefix(shape.rank()-3).elements());
        int outputSize = Math.toIntExact(shape.dimension(shape.rank()-3)*(long)plan.outHeight()*plan.outWidth());
        int offset = Math.toIntExact(call.output().layout().offset());
        boolean backward = call.operation()==Operation.MAX_POOL2D_BACKWARD || call.operation()==Operation.AVG_POOL2D_BACKWARD;
        boolean average = call.operation()==Operation.AVG_POOL2D || call.operation()==Operation.AVG_POOL2D_BACKWARD;
        if (images==0) { return; }
        float[] input = !average || !backward ? CpuTensorKernels.floats(call.inputs().getFirst(),data[0]) : null;
        float[] dy = backward ? CpuTensorKernels.floats(call.inputs().get(1),data[1]) : null;
        int[] indices = backward && !average ? new int[outputSize] : null;
        float[] maxima = indices==null ? null : new float[outputSize];
        for (int image=0;image<images;image++) {
            if (!backward && average) { plan.average(input,image*imageSize,output,offset+image*outputSize); }
            else if (!backward) { plan.max(input,image*imageSize,output,offset+image*outputSize,null,0); }
            else if (average) { plan.averageBackward(dy,image*outputSize,output,offset+image*imageSize); }
            else {
                // Recompute deterministic first-max indices from tape-protected saved inputs.
                plan.max(input,image*imageSize,maxima,0,indices,0);
                plan.maxBackward(indices,0,dy,image*outputSize,output,offset+image*imageSize);
            }
        }
    }
}

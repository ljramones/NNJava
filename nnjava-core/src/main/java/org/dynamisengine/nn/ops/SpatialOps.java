package org.dynamisengine.nn.ops;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;

/** Functional spatial operations with first-order differentiation. */
final class SpatialOps {
    private SpatialOps() { }

    static Tensor conv2d(Tensor input, Tensor weight, Tensor bias, Conv2dConfig config) {
        Objects.requireNonNull(config);
        Shape image = Conv2dConfig.imageShape(input.shape());
        if (weight.shape().rank()!=4 || !weight.shape().equals(config.weightShape(image.dimension(0),weight.shape().dimension(0)))) {
            throw new IllegalArgumentException("Expected convolution weights [outputs,inputs/groups,kernelHeight,kernelWidth]");
        }
        if (bias!=null && !bias.shape().equals(Shape.of(weight.shape().dimension(0)))) { throw new IllegalArgumentException("Convolution bias shape"); }
        List<Tensor> inputs = bias==null ? List.of(input,weight) : List.of(input,weight,bias);
        requireFp32(inputs);
        var attributes = config.attributes().and("bias",bias!=null);
        return Ops.execute(Operation.CONV2D,inputs,config.outputShape(input.shape(),weight.shape().dimension(0)),DType.FP32,0,attributes,
                (dy,saved) -> {
                    var arguments = new ArrayList<>(saved); arguments.add(dy);
                    Tensor[] gradients = new Tensor[saved.size()];
                    try {
                        for (int i=0;i<gradients.length;i++) {
                            if (saved.get(i).requiresGrad()) {
                                gradients[i]=Ops.execute(Operation.CONV2D_BACKWARD,arguments,saved.get(i).shape(),DType.FP32,0,attributes.and("which",i),null);
                            }
                        }
                        return gradients;
                    } catch (RuntimeException | Error error) {
                        for (Tensor gradient : gradients) { if (gradient!=null) { gradient.close(); } }
                        throw error;
                    }
                });
    }

    static Tensor pool2d(Tensor input, Pool2dConfig config, boolean average) {
        Objects.requireNonNull(config); requireFp32(List.of(input));
        var forward = average ? Operation.AVG_POOL2D : Operation.MAX_POOL2D;
        var backward = average ? Operation.AVG_POOL2D_BACKWARD : Operation.MAX_POOL2D_BACKWARD;
        return Ops.execute(forward,List.of(input),config.outputShape(input.shape()),DType.FP32,0,config.attributes(),
                (dy,saved) -> new Tensor[]{Ops.execute(backward,List.of(saved.getFirst(),dy),saved.getFirst().shape(),DType.FP32,0,config.attributes(),null)});
    }

    private static void requireFp32(List<Tensor> tensors) {
        for (var tensor : tensors) { if (tensor.dtype()!=DType.FP32) { throw new IllegalArgumentException("Spatial operations require FP32 tensors"); } }
    }
}

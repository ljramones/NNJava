package org.dynamisengine.nn.nn;

import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;

/**
 * Projection over the last axis; weights are [outputs, inputs].
 */
public final class Linear extends AbstractModule implements UnaryModule {
    private final long inputFeatures;
    private final long outputFeatures;
    private final Parameter weight;
    private final Parameter bias;

    public Linear(ExecutionContext context, long inputs, long outputs, long seed) {
        this(context, inputs, outputs, seed, true);
    }

    public Linear(ExecutionContext context, long inputs, long outputs, long seed, boolean withBias) {
        super(context);
        if (inputs <= 0 || outputs <= 0) {
            throw new IllegalArgumentException("Positive feature counts required");
        }
        inputFeatures = inputs;
        outputFeatures = outputs;
        try (Tensor initial = context.normal(Shape.of(outputs, inputs), seed, (float) Math.sqrt(1.0 / inputs), true)) {
            weight = register("weight", initial);
        }
        try {
            bias = withBias ? constantParameter("bias", Shape.of(outputs), 0) : null;
        } catch (RuntimeException error) {
            close();
            throw error;
        }
    }

    /**
     * Share an existing trainable matrix, retaining independent ownership of its value.
     */
    public Linear(ExecutionContext context, Parameter sharedWeight, boolean withBias) {
        super(context);
        try (Tensor shared = sharedWeight.value()) {
            context.requireContext(shared);
            if (shared.shape().rank() != 2) {
                throw new IllegalArgumentException("Shared weight must be a matrix");
            }
            outputFeatures = shared.shape().dimension(0);
            inputFeatures = shared.shape().dimension(1);
            if (outputFeatures == 0 || inputFeatures == 0) {
                throw new IllegalArgumentException("Empty shared weight");
            }
            weight = register("weight", shared);
        }
        try {
            bias = withBias ? constantParameter("bias", Shape.of(outputFeatures), 0) : null;
        } catch (RuntimeException error) {
            close();
            throw error;
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        ensureOpen();
        context.requireContext(input);
        Shape shape = input.shape();
        if (shape.rank() < 1 || shape.dimension(shape.rank() - 1) != inputFeatures) {
            throw new IllegalArgumentException("Linear last axis must equal " + inputFeatures);
        }
        Shape result = shape.prefix(shape.rank() - 1).append(outputFeatures);
        try (Tensor packed = Ops.contiguous(input); Tensor matrix = Ops.reshape(packed, Shape.of(shape.elements() / inputFeatures, inputFeatures));
             Tensor w = weight.value(); Tensor transposed = Ops.transpose(w); Tensor projected = Ops.matmul(matrix, transposed);
             Tensor shaped = Ops.reshape(projected, result)) {
            if (bias == null) {
                return shaped.retain();
            }
            try (Tensor b = bias.value()) {
                return Ops.add(shaped, b);
            }
        }
    }
}

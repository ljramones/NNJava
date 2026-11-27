package org.dynamisengine.nn.nn;

import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Tensor;

/**
 * Owns and composes layers in registration order.
 */
public final class Sequential extends AbstractModule implements UnaryModule {
    private final UnaryModule[] layers;

    public Sequential(ExecutionContext context, UnaryModule... layers) {
        super(context);
        this.layers = layers.clone();
        for (int i = 0; i < layers.length; i++) {
            register(Integer.toString(i), layers[i]);
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        ensureOpen();
        context.requireContext(input);
        Tensor value = input.retain();
        try {
            for (UnaryModule layer : layers) {
                Tensor next = layer.forward(value);
                value.close();
                value = next;
            }
            return value;
        } catch (RuntimeException error) {
            value.close();
            throw error;
        }
    }
}

package org.dynamisengine.nn.ops;

import org.dynamisengine.nn.backend.KernelAttributes;
import org.dynamisengine.nn.tensor.Shape;

/** Square pooling windows, floor output sizing, and symmetric padding. */
public record Pool2dConfig(int window, int stride, int padding) {
    public Pool2dConfig {
        if (window < 1 || stride < 1 || padding < 0 || 2L*padding > window) {
            throw new IllegalArgumentException("Pooling needs a positive window/stride and padding <= window/2");
        }
    }
    public Pool2dConfig(int window) { this(window,window,0); }
    public Pool2dConfig stride(int value) { return new Pool2dConfig(window,value,padding); }
    public Pool2dConfig padding(int value) { return new Pool2dConfig(window,stride,value); }
    public Shape outputShape(Shape input) {
        Shape image = Conv2dConfig.imageShape(input);
        Shape output = Shape.of(image.dimension(0),Conv2dConfig.outputSize(image.dimension(1),window,stride,padding,1),
                Conv2dConfig.outputSize(image.dimension(2),window,stride,padding,1));
        return input.prefix(input.rank()-3).append(output.dimensions());
    }
    public KernelAttributes attributes() { return KernelAttributes.of("window",window).and("stride",stride).and("padding",padding); }
    public static Pool2dConfig fromAttributes(KernelAttributes a) { return new Pool2dConfig(a.intValue("window"),a.intValue("stride"),a.intValue("padding")); }
}

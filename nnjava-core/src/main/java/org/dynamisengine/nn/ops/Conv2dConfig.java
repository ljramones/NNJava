package org.dynamisengine.nn.ops;

import org.dynamisengine.nn.backend.KernelAttributes;
import org.dynamisengine.nn.tensor.Shape;

/** Channel-first cross-correlation geometry. Padding is symmetric on each axis. */
public record Conv2dConfig(int kernelHeight, int kernelWidth, int strideHeight, int strideWidth,
                           int paddingHeight, int paddingWidth, int dilationHeight, int dilationWidth, int groups) {
    public Conv2dConfig {
        if (kernelHeight < 1 || kernelWidth < 1 || strideHeight < 1 || strideWidth < 1 || paddingHeight < 0 || paddingWidth < 0
                || dilationHeight < 1 || dilationWidth < 1 || groups < 1) {
            throw new IllegalArgumentException("Invalid convolution kernel, stride, padding, dilation, or groups");
        }
    }

    public Conv2dConfig(int kernelHeight, int kernelWidth) { this(kernelHeight,kernelWidth,1,1,0,0,1,1,1); }
    public Conv2dConfig stride(int value) { return stride(value,value); }
    public Conv2dConfig stride(int height, int width) { return new Conv2dConfig(kernelHeight,kernelWidth,height,width,paddingHeight,paddingWidth,dilationHeight,dilationWidth,groups); }
    public Conv2dConfig padding(int value) { return padding(value,value); }
    public Conv2dConfig padding(int height, int width) { return new Conv2dConfig(kernelHeight,kernelWidth,strideHeight,strideWidth,height,width,dilationHeight,dilationWidth,groups); }
    public Conv2dConfig dilation(int value) { return dilation(value,value); }
    public Conv2dConfig dilation(int height, int width) { return new Conv2dConfig(kernelHeight,kernelWidth,strideHeight,strideWidth,paddingHeight,paddingWidth,height,width,groups); }
    public Conv2dConfig groups(int value) { return new Conv2dConfig(kernelHeight,kernelWidth,strideHeight,strideWidth,paddingHeight,paddingWidth,dilationHeight,dilationWidth,value); }

    public Shape weightShape(long inputChannels, long outputChannels) {
        if (inputChannels < 1 || outputChannels < 1 || inputChannels % groups != 0 || outputChannels % groups != 0) {
            throw new IllegalArgumentException("Convolution channels must be positive and divisible by groups");
        }
        return Shape.of(outputChannels,inputChannels/groups,kernelHeight,kernelWidth);
    }

    /** Preserves every leading batch axis, including empty batch dimensions. */
    public Shape outputShape(Shape input, long outputChannels) {
        Shape image = imageShape(input);
        weightShape(image.dimension(0),outputChannels);
        Shape output = Shape.of(outputChannels,outputSize(image.dimension(1),kernelHeight,strideHeight,paddingHeight,dilationHeight),
                outputSize(image.dimension(2),kernelWidth,strideWidth,paddingWidth,dilationWidth));
        return input.prefix(input.rank()-3).append(output.dimensions());
    }

    static long outputSize(long size, int kernel, int stride, int padding, int dilation) {
        long padded = Math.addExact(size,2L*padding);
        long effective = Math.addExact(Math.multiplyExact((long)dilation,kernel-1L),1);
        long output = Math.addExact(Math.floorDiv(Math.subtractExact(padded,effective),stride),1);
        if (output < 1) { throw new IllegalArgumentException("Spatial kernel does not fit the padded input"); }
        return output;
    }

    public static Shape imageShape(Shape input) {
        if (input.rank() < 3) { throw new IllegalArgumentException("Expected trailing [channels,height,width] axes"); }
        int rank = input.rank();
        Shape image = Shape.of(input.dimension(rank-3),input.dimension(rank-2),input.dimension(rank-1));
        for (long size : image.dimensions()) { if (size < 1) { throw new IllegalArgumentException("Spatial dimensions must be positive"); } }
        return image;
    }

    public KernelAttributes attributes() {
        return KernelAttributes.of("kernelH",kernelHeight).and("kernelW",kernelWidth).and("strideH",strideHeight).and("strideW",strideWidth)
                .and("padH",paddingHeight).and("padW",paddingWidth).and("dilationH",dilationHeight).and("dilationW",dilationWidth).and("groups",groups);
    }

    public static Conv2dConfig fromAttributes(KernelAttributes a) {
        return new Conv2dConfig(a.intValue("kernelH"),a.intValue("kernelW"),a.intValue("strideH"),a.intValue("strideW"),
                a.intValue("padH"),a.intValue("padW"),a.intValue("dilationH"),a.intValue("dilationW"),a.intValue("groups"));
    }
}

# Convolutional networks

NNJava's FP32 CPU convolution and pooling use **Vectrix 4.1.0 `Conv2d` and
`Pool2d`**, including their backward kernels. The adapter packs strided tensors
and batches image calls; the numerical implementations remain in Vectrix.

## Build a CNN

```java
var definition = NeuralNetwork.builder()
        .name("image-classifier").inputShape(1, 28, 28).seed(42)
        .conv2d("edges", Layers.conv2d(8, 3)
                .padding(1).activation(Activation.Kind.RELU))
        .maxPool2d("downsample", 2)
        .conv2d("features", 16, 3, Activation.Kind.RELU)
        .avgPool2d("average", 2)
        .flatten("flat").dense("logits", 10)
        .define();
try (var model = definition.build(context)) {
    // FP32 images [batch,1,28,28] produce logits [batch,10].
}
```

Imports come from `org.dynamisengine.nn.nn`. Spatial layer specifications require
a sample shape `[channels,height,width]`. Leading batch axes represent independent
images; functional operations also accept an unbatched `[C,H,W]` image or multiple
leading axes. Channels/spatial dimensions must be positive; empty batches work.

`Layers.conv2d(outputs, kernelHeight, kernelWidth)` returns immutable settings.
Use `.stride(h,w)`, `.padding(h,w)`, `.dilation(h,w)`, `.groups(count)`,
`.bias(false)`, and `.activation(kind)`. Single-argument stride/padding/dilation
set both axes. Defaults are stride 1, padding 0, dilation 1, groups 1, bias enabled,
and no activation. Both channel counts must be divisible by groups. Depthwise
convolution uses `groups == inputChannels` with a divisible output count.

`Conv2d.builder()` supplies a standalone definition with explicit `inputChannels`,
`outputChannels`, `kernelSize`, geometry, bias, and seed. Weights use normal
initialization with standard deviation `sqrt(1/fanIn)`; biases start at zero.
Network seeds derive from the root seed and stable names. Parameter paths include
`edges.conv.weight` and `edges.conv.bias`.

## Operation semantics

Convolution is cross-correlation; kernels are not reversed. Weights have shape
`[outputChannels,inputChannels/groups,kernelHeight,kernelWidth]`. Each output axis is:

```text
floor((input + 2*padding - dilation*(kernel-1) - 1) / stride) + 1
```

Padding is symmetric, independently configurable for height/width. Shape arithmetic
is checked; kernels that do not fit are rejected.

`Layers.maxPool2d(window)` and `Layers.avgPool2d(window)` use square windows,
stride equal to the window, and padding 0. Customize `.stride(value)` and
`.padding(value)` through `.layer(name, specification)`. Sizing uses floor division;
padding must be no greater than `window/2`.

Max pooling excludes padding and chooses the first row-major maximum among ties;
the first NaN wins. Backward accumulates overlapping gradients at that input.
Average pooling excludes padding from its denominator and distributes gradients
over in-bounds cells. Max backward recomputes indices from tape-protected inputs.
Convolution weight/bias gradients accumulate in deterministic image order.

Functional APIs live in `org.dynamisengine.nn.ops`:

```java
var geometry = new Conv2dConfig(3, 3).padding(1).groups(2);
try (var output = Ops.conv2d(input, weights, bias, geometry);
        var pooled = Ops.maxPool2d(output, new Pool2dConfig(2))) {
    // bias may be null; returned tensors are owned handles.
}
```

`Ops.avgPool2d` and the parameterless `Pool2d` module provide average pooling.
Create differentiable views inside the gradient-tape scope. First-order gradients
support strided inputs/weights, shared uses, grouped/depthwise convolution,
optional bias, and empty batches.

## Training and verification

Use the ordinary [Trainer](training-guide.md), objectives, and data loaders.
Definitions round-trip through `ModelDefinitions`; model-only archives and full
training checkpoints retain their existing roles. Tests verify byte-identical
final archives after interrupted and resumed CNN training.

```sh
mvn install
mvn -f nnjava-examples/pom.xml exec:java -Dexec.args=cnn
```

The example uses two convolution/pooling stages and a dense head to classify noisy
vertical/horizontal bars. It checks held-out data, definition restore, and exact
checkpoint continuation. The tiny generated fixture runs without downloads and
establishes correctness rather than image-benchmark performance.

These layers currently execute on CPU. GPU convolution/pooling remain upstream
gaps G-E4/G-T8; unsupported providers fail explicitly. Rectangular pooling, ceil
mode, included-pad averages, channels-last layouts, and transposed convolution
are additional scope. CPU arrays and convolution scratch must fit Vectrix's
int-indexed limits; plan limits are checked before kernel work is accepted.
The initial adapter allocates scratch per call and dispatches requested convolution
gradients separately. Scratch reuse and combined backward dispatch remain possible
performance improvements.

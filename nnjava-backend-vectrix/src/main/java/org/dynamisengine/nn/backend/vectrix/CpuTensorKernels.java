package org.dynamisengine.nn.backend.vectrix;

import org.dynamisengine.nn.backend.KernelAttributes;
import org.dynamisengine.nn.backend.KernelCall;
import org.dynamisengine.nn.backend.KernelTensor;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.TensorLayout;
import org.dynamisengine.vectrix.blas.Blasf;
import org.dynamisengine.vectrix.core.ReproducibleMath;

import java.util.Arrays;

/**
 * Layout/indexing kernels and reductions around Vectrix's numeric primitives.
 */
final class CpuTensorKernels {
    private CpuTensorKernels() {
    }

    static void execute(KernelCall call, Object[] data, Object output) {
        Operation op = call.operation();
        if (op.ordinal() >= Operation.RELU.ordinal()) {
            CpuNeuralKernels.execute(call, data, (float[]) output);
            return;
        }
        if (op == Operation.REDUCE_SUM || op == Operation.REDUCE_MAX || op == Operation.ARGMAX || op == Operation.MAX_BACKWARD) {
            reduce(call, data, output);
            return;
        }
        if (op == Operation.GATHER || op == Operation.SCATTER_ADD) {
            gather(call, data, output);
            return;
        }
        if (op == Operation.CONCAT) {
            concat(call, data, output);
            return;
        }
        if (op == Operation.SLICE_BACKWARD) {
            float[] out = (float[]) output;
            Arrays.fill(out, 0f);
            var a = call.attributes();
            TensorLayout slice = TensorLayout.contiguous(call.output().layout().shape()).slice(a.intValue("axis"),
                    a.integer("start"), a.integer("end"), a.integer("step"));
            float[] values = floats(call.inputs().getFirst(), data[0]);
            for (int i = 0; i < values.length; i++) {
                out[Math.toIntExact(call.output().layout().offset() + slice.index(i))] = values[i];
            }
            return;
        }
        if (op == Operation.NORM2) {
            float[] values = floats(call.inputs().getFirst(), data[0]);
            ((float[]) output)[Math.toIntExact(call.output().layout().offset())] = Blasf.nrm2(values, 0, values.length);
            return;
        }
        if (op == Operation.ALL) {
            boolean all = true;
            for (long i = 0; i < call.inputs().getFirst().layout().shape().elements(); i++) {
                all &= read(data[0], call.inputs().getFirst().layout(), i) != 0;
            }
            ((boolean[]) output)[Math.toIntExact(call.output().layout().offset())] = all;
            return;
        }
        Shape shape = call.output().layout().shape();
        TensorLayout[] layouts = call.inputs().stream().map(i -> i.layout().broadcastTo(shape)).toArray(TensorLayout[]::new);
        for (long i = 0; i < shape.elements(); i++) {
            double a = read(data[0], layouts[0], i);
            double b = data.length > 1 ? read(data[1], layouts[1], i) : 0;
            double value = switch (op) {
                case DIVIDE -> (float) a / (float) b;
                case EXP -> ReproducibleMath.exp((float) a);
                case LOG -> ReproducibleMath.log((float) a);
                case SQRT -> (float) Math.sqrt(a);
                case CAST -> a;
                case EQUAL -> a == b ? 1 : 0;
                case LESS -> a < b ? 1 : 0;
                case GREATER -> a > b ? 1 : 0;
                case AND -> a != 0 && b != 0 ? 1 : 0;
                case OR -> a != 0 || b != 0 ? 1 : 0;
                case NOT -> a == 0 ? 1 : 0;
                case IS_FINITE -> Double.isFinite(a) ? 1 : 0;
                case WHERE -> a != 0 ? b : read(data[2], layouts[2], i);
                default -> throw new UnsupportedOperationException("No CPU kernel for " + op);
            };
            write(output, call.output().layout().index(i), value);
        }
    }

    private static void reduce(KernelCall call, Object[] data, Object output) {
        KernelAttributes attrs = call.attributes();
        Shape inputShape = call.inputs().getFirst().layout().shape();
        Shape groups = KernelCall.reducedShape(inputShape, attrs.axes(), true);
        int count = Math.toIntExact(groups.elements());
        if (count == 0) {
            return;
        }
        float[] values = floats(call.inputs().getFirst(), data[0]);
        TensorLayout destinations = TensorLayout.contiguous(groups).broadcastTo(inputShape);
        int width = values.length / count;
        int[] sizes = new int[count];
        float[] packed = new float[values.length];
        int[] original = new int[values.length];
        for (int i = 0; i < values.length; i++) {
            int group = Math.toIntExact(destinations.index(i));
            int slot = group * width + sizes[group]++;
            packed[slot] = values[i];
            original[slot] = i;
        }
        boolean backward = call.operation() == Operation.MAX_BACKWARD;
        float[] dy = backward ? floats(call.inputs().get(1), data[1]) : null;
        if (backward) {
            Arrays.fill((float[]) output, 0f);
        }
        for (int group = 0; group < count; group++) {
            int start = group * width;
            if (call.operation() == Operation.REDUCE_SUM) {
                write(output, call.output().layout().index(group), Blasf.sum(packed, start, width));
            } else {
                if (width == 0) {
                    throw new IllegalArgumentException("Empty maximum domain");
                }
                int best = start;
                for (int j = start + 1; j < start + width; j++) {
                    if (!Float.isNaN(packed[best]) && (Float.isNaN(packed[j]) || packed[j] > packed[best])) {
                        best = j;
                    }
                }
                if (backward) {
                    ((float[]) output)[Math.toIntExact(call.output().layout().offset()) + original[best]] = dy[group];
                } else {
                    write(output, call.output().layout().index(group), call.operation() == Operation.ARGMAX ? best - start : packed[best]);
                }
            }
        }
    }

    private static void gather(KernelCall call, Object[] data, Object output) {
        boolean scatter = call.operation() == Operation.SCATTER_ADD;
        Shape source = scatter ? call.output().layout().shape() : call.inputs().getFirst().layout().shape();
        Shape indicesShape = call.inputs().get(1).layout().shape();
        int axis = source.axis(call.attributes().intValue("axis"));
        int[] indices = integers(call.inputs().get(1), data[1]);
        for (int index : indices) {
            if (index < 0 || index >= source.dimension(axis)) {
                throw new IllegalArgumentException("Gather index outside axis: " + index);
            }
        }
        Shape expanded = KernelCall.gatherShape(source, indicesShape, axis);
        if (scatter) {
            Arrays.fill((float[]) output, 0f);
        }
        for (long i = 0; i < expanded.elements(); i++) {
            long[] coordinates = coordinates(expanded, i);
            long index = 0;
            for (int j = 0; j < indicesShape.rank(); j++) {
                index = index * indicesShape.dimension(j) + coordinates[axis + j];
            }
            long[] original = new long[source.rank()];
            System.arraycopy(coordinates, 0, original, 0, axis);
            original[axis] = indices[Math.toIntExact(index)];
            for (int j = axis + 1; j < source.rank(); j++) {
                original[j] = coordinates[j + indicesShape.rank() - 1];
            }
            long logical = linear(source, original);
            if (scatter) {
                int destination = Math.toIntExact(call.output().layout().index(logical));
                ((float[]) output)[destination] += (float) read(data[0], call.inputs().getFirst().layout(), i);
            } else {
                write(output, call.output().layout().index(i), read(data[0], call.inputs().getFirst().layout(), logical));
            }
        }
    }

    private static void concat(KernelCall call, Object[] data, Object output) {
        Shape shape = call.output().layout().shape();
        int axis = shape.axis(call.attributes().intValue("axis"));
        for (long i = 0; i < shape.elements(); i++) {
            long[] coordinates = coordinates(shape, i);
            int selected = 0;
            while (coordinates[axis] >= call.inputs().get(selected).layout().shape().dimension(axis)) {
                coordinates[axis] -= call.inputs().get(selected).layout().shape().dimension(axis);
                selected++;
            }
            KernelTensor input = call.inputs().get(selected);
            write(output, call.output().layout().index(i), read(data[selected], input.layout(), linear(input.layout().shape(), coordinates)));
        }
    }

    static float[] floats(KernelTensor tensor, Object array) {
        int count = Math.toIntExact(tensor.layout().shape().elements());
        float[] result = new float[count];
        for (int i = 0; i < count; i++) {
            result[i] = (float) read(array, tensor.layout(), i);
        }
        return result;
    }

    static int[] integers(KernelTensor tensor, Object array) {
        int[] result = new int[Math.toIntExact(tensor.layout().shape().elements())];
        for (int i = 0; i < result.length; i++) {
            result[i] = (int) read(array, tensor.layout(), i);
        }
        return result;
    }

    static double read(Object array, TensorLayout layout, long logical) {
        int index = Math.toIntExact(layout.index(logical));
        if (array instanceof float[] values) {
            return values[index];
        }
        if (array instanceof int[] values) {
            return values[index];
        }
        return ((boolean[]) array)[index] ? 1 : 0;
    }

    static void write(Object array, long physical, double value) {
        int index = Math.toIntExact(physical);
        if (array instanceof float[] values) {
            values[index] = (float) value;
        } else if (array instanceof int[] values) {
            values[index] = (int) value;
        } else {
            ((boolean[]) array)[index] = value != 0;
        }
    }

    static long[] coordinates(Shape shape, long linear) {
        long[] result = new long[shape.rank()];
        for (int i = shape.rank() - 1; i >= 0; i--) {
            result[i] = linear % shape.dimension(i);
            linear /= shape.dimension(i);
        }
        return result;
    }

    static long linear(Shape shape, long[] coordinates) {
        long result = 0;
        for (int i = 0; i < shape.rank(); i++) {
            result = result * shape.dimension(i) + coordinates[i];
        }
        return result;
    }
}

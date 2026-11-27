package org.dynamisengine.nn.backend.vectrix;

import org.dynamisengine.nn.backend.KernelCall;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.TensorLayout;
import org.dynamisengine.vectrix.nn.*;

import java.util.Arrays;

/**
 * Vectrix neural kernels packed from arbitrary supported tensor layouts.
 */
final class CpuNeuralKernels {
    private CpuNeuralKernels() {
    }

    static void execute(KernelCall call, Object[] data, float[] output) {
        Operation op = call.operation();
        if (op == Operation.LAYER_NORM || op == Operation.RMS_NORM || op == Operation.NORM_BACKWARD) {
            normalize(call, data, output);
            return;
        }
        if (op == Operation.CROSS_ENTROPY || op == Operation.CROSS_ENTROPY_BACKWARD) {
            loss(call, data, output);
            return;
        }
        float[] x = CpuTensorKernels.floats(call.inputs().getFirst(), data[0]);
        int offset = Math.toIntExact(call.output().layout().offset());
        int n = x.length;
        if (op == Operation.ROPE || op == Operation.ROPE_BACKWARD) {
            int width = Math.toIntExact(call.inputs().getFirst().layout().shape().dimension(call.inputs().getFirst().layout().shape().rank() - 1));
            Shape rows = call.inputs().getFirst().layout().shape().prefix(call.inputs().getFirst().layout().shape().rank() - 1);
            long sequence = rows.dimension(rows.rank() - 1);
            var attrs = call.attributes();
            Rope plan = Rope.plan(width, attrs.number("base"), attrs.flag("half") ? Rope.Layout.HALF : Rope.Layout.INTERLEAVED);
            TensorLayout positions = data.length == 2 ? call.inputs().get(1).layout().broadcastTo(rows) : null;
            if (positions != null) {
                for (long row = 0; row < rows.elements(); row++) {
                    if (CpuTensorKernels.read(data[1], positions, row) < 0) {
                        throw new IllegalArgumentException("Negative position");
                    }
                }
            }
            for (int row = 0; row < rows.elements(); row++) {
                long position = positions == null ? attrs.integer("offset") + row % sequence : (long) CpuTensorKernels.read(data[1], positions, row);
                if (op == Operation.ROPE) {
                    plan.apply(x, row * width, position);
                } else {
                    plan.applyBackward(x, row * width, position);
                }
            }
            System.arraycopy(x, 0, output, offset, n);
            return;
        }
        if (op == Operation.DROPOUT) {
            var a = call.attributes();
            Activations.dropout(a.integer("seed"), a.integer("counter"), (float) a.number("p"), x, 0, output, offset, n);
            return;
        }
        boolean backward = op.name().endsWith("_BACKWARD");
        float[] dy = backward ? CpuTensorKernels.floats(call.inputs().get(1), data[1]) : null;
        if (op == Operation.SOFTMAX || op == Operation.LOG_SOFTMAX || op == Operation.SOFTMAX_BACKWARD || op == Operation.LOG_SOFTMAX_BACKWARD) {
            int width = Math.toIntExact(call.inputs().getFirst().layout().shape().dimension(call.inputs().getFirst().layout().shape().rank() - 1));
            float[] y = new float[width];
            float[] scratch = new float[width];
            for (int row = 0; row < n; row += width) {
                boolean allMasked = true;
                for (int i = 0; i < width; i++) {
                    allMasked &= x[row + i] == Float.NEGATIVE_INFINITY;
                }
                if (allMasked) {
                    Arrays.fill(output, offset + row, offset + row + width, op == Operation.LOG_SOFTMAX ? Float.NEGATIVE_INFINITY : 0f);
                } else if (op == Operation.SOFTMAX) {
                    Softmax.softmax(x, row, output, offset + row, width);
                } else if (op == Operation.LOG_SOFTMAX) {
                    Softmax.logSoftmax(x, row, output, offset + row, width, scratch, 0);
                } else if (op == Operation.SOFTMAX_BACKWARD) {
                    Softmax.softmax(x, row, y, 0, width);
                    Softmax.softmaxBackward(y, 0, dy, row, output, offset + row, width);
                } else {
                    Softmax.logSoftmax(x, row, y, 0, width, scratch, 0);
                    Softmax.logSoftmaxBackward(y, 0, dy, row, output, offset + row, width);
                }
            }
            return;
        }
        switch (op) {
            case RELU -> Activations.relu(x, 0, output, offset, n);
            case SIGMOID -> Activations.sigmoid(x, 0, output, offset, n);
            case TANH -> Activations.tanh(x, 0, output, offset, n);
            case GELU -> Activations.gelu(x, 0, output, offset, n);
            case SILU -> Activations.silu(x, 0, output, offset, n);
            case RELU_BACKWARD -> Activations.reluBackward(x, 0, dy, 0, output, offset, n);
            case GELU_BACKWARD -> Activations.geluBackward(x, 0, dy, 0, output, offset, n);
            case SILU_BACKWARD -> Activations.siluBackward(x, 0, dy, 0, output, offset, n);
            case SIGMOID_BACKWARD -> {
                Activations.sigmoid(x, 0, x, 0, n);
                Activations.sigmoidBackward(x, 0, dy, 0, output, offset, n);
            }
            case TANH_BACKWARD -> {
                Activations.tanh(x, 0, x, 0, n);
                Activations.tanhBackward(x, 0, dy, 0, output, offset, n);
            }
            default -> throw new UnsupportedOperationException("Missing neural kernel " + op);
        }
    }

    private static void normalize(KernelCall call, Object[] data, float[] output) {
        var attrs = call.attributes();
        boolean rms = call.operation() == Operation.RMS_NORM || call.operation() == Operation.NORM_BACKWARD && attrs.flag("rms");
        boolean backward = call.operation() == Operation.NORM_BACKWARD;
        Shape shape = call.inputs().getFirst().layout().shape();
        int width = Math.toIntExact(shape.dimension(shape.rank() - 1));
        float epsilon = (float) attrs.number("epsilon");
        float[] x = CpuTensorKernels.floats(call.inputs().getFirst(), data[0]);
        float[] scale = CpuTensorKernels.floats(call.inputs().get(1), data[1]);
        float[] bias = rms ? null : CpuTensorKernels.floats(call.inputs().get(2), data[2]);
        float[] dy = backward ? CpuTensorKernels.floats(call.inputs().getLast(), data[data.length - 1]) : null;
        float[] dx = backward ? new float[x.length] : null;
        float[] dg = new float[width];
        float[] db = new float[width];
        float[] stats = new float[2];
        float[] temporary = new float[width];
        float[] scratch = new float[2 * width];
        int out = Math.toIntExact(call.output().layout().offset());
        for (int row = 0; row < x.length; row += width) {
            if (rms) {
                Norms.rmsNorm(x, row, scale, 0, epsilon, backward ? temporary : output, backward ? 0 : out + row, width, stats, 0);
            } else {
                Norms.layerNorm(x, row, scale, 0, bias, 0, epsilon, backward ? temporary : output, backward ? 0 : out + row, width, stats, 0);
            }
            if (backward) {
                if (rms) {
                    Norms.rmsNormBackward(x, row, scale, 0, stats, 0, dy, row, dx, row, dg, 0, width, scratch, 0);
                } else {
                    Norms.layerNormBackward(x, row, scale, 0, stats, 0, dy, row, dx, row, dg, 0, db, 0, width, scratch, 0);
                }
            }
        }
        if (backward) {
            float[] result = switch (attrs.intValue("which")) {
                case 0 -> dx;
                case 1 -> dg;
                case 2 -> db;
                default -> throw new IllegalArgumentException("Normalization gradient index");
            };
            System.arraycopy(result, 0, output, out, result.length);
        }
    }

    private static void loss(KernelCall call, Object[] data, float[] output) {
        Shape shape = call.inputs().getFirst().layout().shape();
        int width = Math.toIntExact(shape.dimension(shape.rank() - 1));
        float[] logits = CpuTensorKernels.floats(call.inputs().getFirst(), data[0]);
        int[] targets = CpuTensorKernels.integers(call.inputs().get(1), data[1]);
        var attrs = call.attributes();
        String reduction = attrs.text("reduction");
        int ignored = attrs.intValue("ignore");
        int valid = 0;
        for (int target : targets) {
            if (target != ignored) {
                if (target < 0 || target >= width) {
                    throw new IllegalArgumentException("Cross-entropy target outside vocabulary");
                }
                valid++;
            }
        }
        int offset = Math.toIntExact(call.output().layout().offset());
        Arrays.fill(output, offset, offset + Math.toIntExact(call.output().layout().shape().elements()), 0f);
        float[] scratch = new float[width];
        float[] dy = call.operation() == Operation.CROSS_ENTROPY_BACKWARD ? CpuTensorKernels.floats(call.inputs().get(2), data[2]) : null;
        for (int row = 0; row < targets.length; row++) {
            if (targets[row] == ignored) {
                continue;
            }
            float scale = reduction.equals("mean") ? 1f / valid : 1f;
            if (dy == null) {
                float value = Loss.crossEntropy(logits, row * width, width, targets[row], scratch, 0) * scale;
                if (reduction.equals("none")) {
                    output[offset + row] = value;
                } else {
                    output[offset] += value;
                }
            } else {
                Loss.crossEntropyBackward(logits, row * width, width, targets[row], scale * dy[reduction.equals("none") ? row : 0], output, offset + row * width);
            }
        }
    }
}

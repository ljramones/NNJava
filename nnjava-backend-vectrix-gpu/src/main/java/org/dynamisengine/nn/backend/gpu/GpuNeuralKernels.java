package org.dynamisengine.nn.backend.gpu;

import java.util.Arrays;
import java.util.List;
import org.dynamisengine.gpu.api.compute.GpuCommandEncoder;
import org.dynamisengine.gpu.api.compute.GpuDevice;
import org.dynamisengine.gpu.api.compute.kit.AccuracyContract;
import org.dynamisengine.gpu.api.compute.kit.StridedView;
import org.dynamisengine.gpu.primitives.Elementwise;
import org.dynamisengine.gpu.primitives.Logic;
import org.dynamisengine.gpu.primitives.Norms;
import org.dynamisengine.gpu.primitives.Precision;
import org.dynamisengine.gpu.primitives.Reduce;
import org.dynamisengine.gpu.primitives.Softmax;
import org.dynamisengine.nn.backend.KernelCall;
import org.dynamisengine.nn.backend.KernelTensor;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.TensorLayout;
import org.dynamisengine.vectrix.compute.GpuActivations;
import org.dynamisengine.vectrix.compute.GpuBlasf;
import org.dynamisengine.vectrix.compute.GpuNorms;
import org.dynamisengine.vectrix.compute.GpuPointwise;
import org.dynamisengine.vectrix.compute.GpuSoftmax;

/** Device-only neural compositions, keeping NNJava semantics separate from upstream kernel variants. */
final class GpuNeuralKernels implements AutoCloseable {
    private final GpuBackend backend;
    private final GpuDevice device;
    private final GpuMode mode;
    private GpuActivations activations;
    private GpuSoftmax exactSoftmax;
    private GpuNorms exactNorms;
    private GpuBlasf blas;
    private GpuPointwise pointwise;
    private Elementwise elementwise;
    private Softmax softmax;
    private Norms norms;
    private Reduce reduce;
    private Logic logic;

    GpuNeuralKernels(GpuBackend backend, GpuDevice device, GpuMode mode) {
        this.backend = backend; this.device = device; this.mode = mode;
    }
    static boolean handles(Operation op) {
        return switch (op) {
            case RELU, SIGMOID, TANH, GELU, SILU, RELU_BACKWARD, SIGMOID_BACKWARD, TANH_BACKWARD,
                    GELU_BACKWARD, SILU_BACKWARD, SOFTMAX, LOG_SOFTMAX, SOFTMAX_BACKWARD, LOG_SOFTMAX_BACKWARD,
                    LAYER_NORM, RMS_NORM, NORM_BACKWARD, SUM, SUM_TO, REDUCE_SUM, NORM2, DROPOUT -> true;
            default -> false;
        };
    }
    static boolean rowOperation(Operation op) {
        return switch (op) {
            case SOFTMAX, LOG_SOFTMAX, SOFTMAX_BACKWARD, LOG_SOFTMAX_BACKWARD, LAYER_NORM, RMS_NORM, NORM_BACKWARD -> true;
            default -> false;
        };
    }
    static boolean supports(KernelCall call) {
        if (call.output().buffer().dtype() != DType.FP32 || !call.output().layout().isContiguous()
                || call.inputs().stream().anyMatch(t -> t.buffer().dtype() != DType.FP32)) { return false; }
        if (rowOperation(call.operation())) {
            var shape = call.inputs().getFirst().layout().shape();
            return shape.rank() > 0 && shape.dimension(shape.rank() - 1) > 0
                    && shape.dimension(shape.rank() - 1) <= GpuSoftmax.MAX_ROW;
        }
        return true;
    }
    void encode(GpuCommandEncoder encoder, KernelCall call) {
        try (var scratch = new GpuScratch(backend)) {
            switch (call.operation()) {
                case SOFTMAX, LOG_SOFTMAX, SOFTMAX_BACKWARD, LOG_SOFTMAX_BACKWARD -> softmax(encoder, call, scratch);
                case LAYER_NORM, RMS_NORM, NORM_BACKWARD -> normalize(encoder, call, scratch);
                case SUM, SUM_TO, REDUCE_SUM, NORM2 -> reduce(encoder, call, scratch);
                default -> activation(encoder, call, scratch);
            }
        }
    }
    private GpuActivations activations() { if (activations == null) { activations = new GpuActivations(device); } return activations; }
    private GpuSoftmax exactSoftmax() { if (exactSoftmax == null) { exactSoftmax = new GpuSoftmax(device); } return exactSoftmax; }
    private GpuNorms exactNorms() { if (exactNorms == null) { exactNorms = new GpuNorms(device); } return exactNorms; }
    private GpuBlasf blas() { if (blas == null) { blas = new GpuBlasf(device); } return blas; }
    private Reduce reductions() { if (reduce == null) { reduce = Reduce.create(device); } return reduce; }
    private Logic logic() { if (logic == null) { logic = Logic.create(device); } return logic; }
    private Elementwise elementwise() { if (elementwise == null) { elementwise = Elementwise.create(device); } return elementwise; }
    private static StridedView vector(KernelTensor tensor) {
        return StridedView.of(tensor.layout().offset(), new long[]{tensor.layout().shape().elements()}, new long[]{1});
    }
    private void selected(Operation operation, String tier, AccuracyContract contract) { backend.selected(operation, tier, contract); }
    private void exact(Operation op) { selected(op, "vectrix-gpu", exactContract(op)); }
    static AccuracyContract exactContract(Operation op) {
        return new AccuracyContract(op.name(), List.of("FP32"), "FP32, Vectrix/NNJava CPU order", "FP32",
                AccuracyContract.Order.FIXED, AccuracyContract.Bound.exact(), "NNJava CPU " + op + " using Vectrix 4.1.0");
    }

    private void activation(GpuCommandEncoder e, KernelCall call, GpuScratch scratch) {
        var op = call.operation(); var x = scratch.packed(e, call.inputs().getFirst(), false); var out = call.output();
        int n = Math.toIntExact(out.layout().shape().elements()); long xo = x.layout().offset(), oo = out.layout().offset();
        boolean backward = call.inputs().size() == 2;
        if (op == Operation.DROPOUT) {
            var attributes = call.attributes();
            activations().dropout(e, attributes.integer("seed"), attributes.integer("counter"), (float) attributes.number("p"),
                    backend.storage(x), xo, backend.storage(out), oo, n);
            exact(op); return;
        }
        var dy = backward ? scratch.packed(e, call.inputs().get(1), false) : null;
        // NNJava GELU is erf-based. DynamisGPU GELU is tanh-based; RELU also has different -0 semantics.
        boolean twin = mode == GpuMode.EXACT || op == Operation.GELU || op == Operation.GELU_BACKWARD
                || op == Operation.RELU || op == Operation.RELU_BACKWARD;
        if (!twin) {
            if (backward) {
                var operation = Elementwise.Binary.valueOf(op.name());
                elementwise().binary(e, operation, backend.storage(x), vector(x), backend.storage(dy), vector(dy), backend.storage(out), vector(out));
                selected(op, "dynamis-gpu-primitives", Elementwise.contract(operation));
            } else {
                var operation = Elementwise.Unary.valueOf(op.name());
                elementwise().unary(e, operation, backend.storage(x), vector(x), backend.storage(out), vector(out));
                selected(op, "dynamis-gpu-primitives", Elementwise.contract(operation));
            }
            return;
        }
        var a = activations();
        if (!backward) {
            switch (op) {
                case RELU -> a.relu(e, backend.storage(x), xo, backend.storage(out), oo, n);
                case SIGMOID -> a.sigmoid(e, backend.storage(x), xo, backend.storage(out), oo, n);
                case TANH -> a.tanh(e, backend.storage(x), xo, backend.storage(out), oo, n);
                case SILU -> a.silu(e, backend.storage(x), xo, backend.storage(out), oo, n);
                case GELU -> a.gelu(e, backend.storage(x), xo, backend.storage(out), oo, n);
                default -> throw new UnsupportedOperationException(op.toString());
            }
        } else {
            var argument = x;
            if (op == Operation.SIGMOID_BACKWARD || op == Operation.TANH_BACKWARD) {
                argument = scratch.tensor(x.layout().shape());
                if (op == Operation.SIGMOID_BACKWARD) { a.sigmoid(e, backend.storage(x), xo, backend.storage(argument), 0, n); }
                else { a.tanh(e, backend.storage(x), xo, backend.storage(argument), 0, n); }
            }
            long ao = argument.layout().offset(), go = dy.layout().offset();
            switch (op) {
                case RELU_BACKWARD -> a.reluBackward(e, backend.storage(argument), ao, backend.storage(dy), go, backend.storage(out), oo, n);
                case SIGMOID_BACKWARD -> a.sigmoidBackward(e, backend.storage(argument), ao, backend.storage(dy), go, backend.storage(out), oo, n);
                case TANH_BACKWARD -> a.tanhBackward(e, backend.storage(argument), ao, backend.storage(dy), go, backend.storage(out), oo, n);
                case SILU_BACKWARD -> a.siluBackward(e, backend.storage(argument), ao, backend.storage(dy), go, backend.storage(out), oo, n);
                case GELU_BACKWARD -> a.geluBackward(e, backend.storage(argument), ao, backend.storage(dy), go, backend.storage(out), oo, n);
                default -> throw new UnsupportedOperationException(op.toString());
            }
        }
        exact(op);
    }

    private void softmax(GpuCommandEncoder e, KernelCall call, GpuScratch scratch) {
        var op = call.operation(); var x = scratch.packed(e, call.inputs().getFirst(), true);
        var shape = x.layout().shape(); int n = Math.toIntExact(shape.dimension(shape.rank() - 1)), rows = Math.toIntExact(shape.elements() / n);
        boolean logarithmic = op == Operation.LOG_SOFTMAX || op == Operation.LOG_SOFTMAX_BACKWARD;
        boolean backward = call.inputs().size() == 2; boolean twin = mode == GpuMode.EXACT || logarithmic;
        var y = scratch.tensor(shape); var result = backward ? scratch.tensor(shape) : y;
        if (twin) {
            if (logarithmic) { exactSoftmax().logSoftmax(e, rows, backend.storage(x), 0, n, backend.storage(y), 0, n, n); }
            else { exactSoftmax().softmax(e, rows, backend.storage(x), 0, n, backend.storage(y), 0, n, n); }
        } else {
            if (softmax == null) { softmax = Softmax.create(device, Precision.FP32); }
            softmax.encode(e, backend.storage(x), backend.storage(y), Softmax.Shape.of(rows, n));
        }
        if (backward) {
            var dy = scratch.packed(e, call.inputs().get(1), true);
            if (twin) {
                if (logarithmic) { exactSoftmax().logSoftmaxBackward(e, rows, backend.storage(y), 0, n, backend.storage(dy), 0, n, backend.storage(result), 0, n, n); }
                else { exactSoftmax().softmaxBackward(e, rows, backend.storage(y), 0, n, backend.storage(dy), 0, n, backend.storage(result), 0, n, n); }
            } else { softmax.backward(e, backend.storage(y), backend.storage(dy), backend.storage(result), Softmax.Shape.of(rows, n)); }
        }
        // NNJava defines an all--infinity row as probabilities/gradient 0, log probabilities -infinity.
        // Detect on device, then select without propagating a NaN from the unselected kernel result.
        var maxima = scratch.tensor(Shape.of(rows));
        reductions().rows(e, Reduce.Op.MAX, backend.storage(x), StridedView.matrix(rows, n), backend.storage(maxima), StridedView.contiguous(rows));
        var negative = scratch.tensor(Shape.of(1)); backend.fill(e, negative, Float.NEGATIVE_INFINITY);
        var masked = scratch.tensor(DType.BOOL, Shape.of(rows));
        logic().compare(e, Logic.Compare.EQUAL, Logic.Operand.FP32, backend.storage(maxima), StridedView.contiguous(rows),
                backend.storage(negative), StridedView.contiguous(1), backend.storage(masked), StridedView.contiguous(rows));
        var replacement = scratch.tensor(Shape.of(1)); backend.fill(e, replacement, logarithmic && !backward ? Float.NEGATIVE_INFINITY : 0);
        logic().select(e, backend.storage(masked), StridedView.of(0, new long[]{rows, 1}, new long[]{1, 1}),
                backend.storage(replacement), StridedView.contiguous(1), backend.storage(result), StridedView.matrix(rows, n),
                backend.storage(call.output()), StridedView.of(call.output().layout().offset(), new long[]{rows, n}, new long[]{n, 1}));
        if (twin) { exact(op); }
        else { selected(op, "dynamis-gpu-primitives", backward ? Softmax.backwardContract() : Softmax.contract(Precision.FP32)); }
    }

    private void normalize(GpuCommandEncoder e, KernelCall call, GpuScratch scratch) {
        var op = call.operation(); boolean backward = op == Operation.NORM_BACKWARD;
        boolean rms = op == Operation.RMS_NORM || backward && call.attributes().flag("rms");
        int which = backward ? call.attributes().intValue("which") : 0;
        var x = scratch.packed(e, call.inputs().getFirst(), true); var weight = scratch.packed(e, call.inputs().get(1), true);
        var bias = rms ? null : scratch.packed(e, call.inputs().get(2), true);
        var shape = x.layout().shape(); int n = Math.toIntExact(shape.dimension(shape.rank() - 1)), rows = Math.toIntExact(shape.elements() / n);
        float eps = (float) call.attributes().number("epsilon");
        var out = call.output().layout().offset() == 0 ? call.output() : scratch.tensor(call.output().layout().shape());
        if (rows == 0) { backend.fill(e, out, 0); if (out != call.output()) { backend.copy(e, out, call.output()); } exact(op); return; }
        if (mode == GpuMode.EXACT) {
            var stats = backward ? scratch.tensor(Shape.of((rms ? 1L : 2L) * rows)) : null;
            var y = backward ? scratch.tensor(shape) : out;
            if (rms) { exactNorms().rmsNorm(e, rows, backend.storage(x), 0, n, backend.storage(weight), 0, eps, backend.storage(y), 0, n, n, stats == null ? null : backend.storage(stats), 0); }
            else { exactNorms().layerNorm(e, rows, backend.storage(x), 0, n, backend.storage(weight), 0, backend.storage(bias), 0, eps, backend.storage(y), 0, n, n, stats == null ? null : backend.storage(stats), 0); }
            if (backward) {
                var dy = scratch.packed(e, call.inputs().getLast(), true);
                var dx = which == 0 ? out : scratch.tensor(shape);
                var dg = which == 1 ? out : null; var db = which == 2 ? out : null;
                if (dg != null) { backend.fill(e, dg, 0); } if (db != null) { backend.fill(e, db, 0); }
                if (rms) { exactNorms().rmsNormBackward(e, rows, backend.storage(x), 0, n, backend.storage(weight), 0, backend.storage(stats), 0,
                        backend.storage(dy), 0, n, backend.storage(dx), 0, n, dg == null ? null : backend.storage(dg), 0, n); }
                else { exactNorms().layerNormBackward(e, rows, backend.storage(x), 0, n, backend.storage(weight), 0, backend.storage(stats), 0,
                        backend.storage(dy), 0, n, backend.storage(dx), 0, n, dg == null ? null : backend.storage(dg), 0, db == null ? null : backend.storage(db), 0, n); }
            }
            exact(op);
        } else {
            if (norms == null) { norms = Norms.create(device, Precision.FP32, Precision.FP32); }
            var kind = rms ? Norms.Kind.RMS : Norms.Kind.LAYER;
            if (backward) {
                var dy = scratch.packed(e, call.inputs().getLast(), true); var dx = which == 0 ? out : scratch.tensor(shape);
                var dg = which == 1 ? out : which == 2 ? scratch.tensor(Shape.of(n)) : null; var db = which == 2 ? out : null;
                if (dg != null) { backend.fill(e, dg, 0); } if (db != null) { backend.fill(e, db, 0); }
                norms.backward(e, kind, backend.storage(x), backend.storage(weight), backend.storage(dy), backend.storage(dx),
                        dg == null ? null : backend.storage(dg), db == null ? null : backend.storage(db), Norms.BackwardShape.of(rows, n, eps), false);
                selected(op, "dynamis-gpu-primitives", Norms.backwardContract(kind, Precision.FP32));
            } else {
                norms.encode(e, kind, backend.storage(x), null, null, backend.storage(weight), bias == null ? null : backend.storage(bias),
                        backend.storage(out), Norms.Shape.of(rows, n, eps));
                selected(op, "dynamis-gpu-primitives", Norms.contract(kind, Precision.FP32, Precision.FP32));
            }
        }
        if (out != call.output()) { backend.copy(e, out, call.output()); }
    }

    private void reduce(GpuCommandEncoder e, KernelCall call, GpuScratch scratch) {
        var op = call.operation(); var input = call.inputs().getFirst(); var shape = input.layout().shape();
        if (op == Operation.NORM2) {
            var x = scratch.packed(e, input, false); int n = Math.toIntExact(shape.elements());
            if (n == 0) { backend.fill(e, call.output(), 0); }
            else { var work = scratch.tensor(Shape.of(GpuBlasf.scratchSize(n)));
                blas().nrm2(e, backend.storage(x), x.layout().offset(), n, backend.storage(call.output()), call.output().layout().offset(), backend.storage(work), 0); }
            exact(op); return;
        }
        boolean[] axes = new boolean[shape.rank()];
        if (op == Operation.SUM) { Arrays.fill(axes, true); }
        else if (op == Operation.REDUCE_SUM) { for (int axis : call.attributes().axes()) { axes[axis] = true; } }
        else {
            var target = call.output().layout().shape(); int extra = shape.rank() - target.rank();
            for (int i = 0; i < axes.length; i++) { axes[i] = i < extra || target.dimension(i - extra) == 1; }
        }
        int[] permutation = new int[axes.length]; int slot = 0; long rows = 1, columns = 1;
        for (int i = 0; i < axes.length; i++) { if (!axes[i]) { permutation[slot++] = i; rows *= shape.dimension(i); } }
        for (int i = 0; i < axes.length; i++) { if (axes[i]) { permutation[slot++] = i; columns *= shape.dimension(i); } }
        if (columns == 0) { backend.fill(e, call.output(), 0); exact(op); return; }
        var permuted = new KernelTensor(input.buffer(), input.layout().permute(permutation));
        var x = GpuScratch.flat(scratch.packed(e, permuted, true), Shape.of(rows, columns));
        int count = Math.toIntExact(rows), width = Math.toIntExact(columns);
        if (mode == GpuMode.THROUGHPUT) {
            reductions().rows(e, Reduce.Op.SUM, backend.storage(x), StridedView.matrix(rows, columns), backend.storage(call.output()),
                    StridedView.of(call.output().layout().offset(), new long[]{rows}, new long[]{1}));
            selected(op, "dynamis-gpu-primitives", Reduce.contract(Reduce.Op.SUM));
        } else if (op == Operation.SUM_TO) {
            // CPU sumTo folds from +0 in input order, unlike Blasf.sum's tree. Preserve each target's order.
            backend.fill(e, call.output(), 0);
            if (pointwise == null) { pointwise = new GpuPointwise(device); }
            var transposed = new KernelTensor(x.buffer(), x.layout().permute(1, 0));
            var values = scratch.packed(e, transposed, true);
            for (int i = 0; i < width; i++) { pointwise.add(e, backend.storage(values), (long) i * count, backend.storage(call.output()),
                    call.output().layout().offset(), backend.storage(call.output()), call.output().layout().offset(), count); }
            exact(op);
        } else {
            var work = scratch.tensor(Shape.of(GpuBlasf.scratchSize(width)));
            for (int row = 0; row < count; row++) { blas().sum(e, backend.storage(x), (long) row * width, width,
                    backend.storage(call.output()), call.output().layout().offset() + row, backend.storage(work), 0); }
            exact(op);
        }
    }

    @Override public void close() throws Exception {
        Exception failure = null;
        for (var resource : new AutoCloseable[]{activations, exactSoftmax, exactNorms, blas, pointwise, elementwise, softmax, norms, reduce, logic}) {
            if (resource != null) { try { resource.close(); } catch (Exception error) { if (failure == null) { failure = error; } else { failure.addSuppressed(error); } } }
        }
        if (failure != null) { throw failure; }
    }
}

package org.dynamisengine.nn.backend.gpu;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.dynamisengine.gpu.api.compute.GpuCommandEncoder;
import org.dynamisengine.gpu.api.compute.GpuDevice;
import org.dynamisengine.gpu.api.compute.kit.AccuracyContract;
import org.dynamisengine.gpu.api.compute.kit.StridedView;
import org.dynamisengine.gpu.primitives.Cast;
import org.dynamisengine.gpu.primitives.Gather;
import org.dynamisengine.gpu.primitives.Logic;
import org.dynamisengine.gpu.primitives.Reduce;
import org.dynamisengine.nn.backend.KernelCall;
import org.dynamisengine.nn.backend.KernelTensor;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.vectrix.compute.GpuRope;
import org.dynamisengine.vectrix.nn.Rope;

/** Exact word/index operations and CPU-compatible rotary embeddings, composed entirely on the device. */
final class GpuTensorKernels implements AutoCloseable {
    private final GpuBackend backend;
    private final GpuDevice device;
    private Logic logic;
    private Cast cast;
    private Gather gather;
    private Reduce reduce;
    private GpuRope rope;

    GpuTensorKernels(GpuBackend backend, GpuDevice device) { this.backend = backend; this.device = device; }

    static boolean handles(Operation operation) {
        return switch (operation) {
            case CAST, EQUAL, LESS, GREATER, AND, OR, NOT, WHERE, IS_FINITE, ALL,
                    GATHER, SCATTER_ADD, CONCAT, SLICE_BACKWARD, REDUCE_MAX, ARGMAX, MAX_BACKWARD,
                    ROPE, ROPE_BACKWARD -> true;
            default -> false;
        };
    }

    private static boolean wordType(DType dtype) { return dtype == DType.FP32 || dtype == DType.INT32 || dtype == DType.BOOL; }

    static boolean supports(Operation operation, DType dtype) {
        return switch (operation) {
            case CAST, WHERE, GATHER, CONCAT -> wordType(dtype);
            case EQUAL, LESS, GREATER, AND, OR, NOT, IS_FINITE, ALL -> dtype == DType.BOOL;
            case ARGMAX -> dtype == DType.INT32;
            default -> dtype == DType.FP32;
        };
    }

    static boolean supports(KernelCall call) {
        if (!call.output().layout().isContiguous() || call.inputs().stream().anyMatch(t -> !wordType(t.buffer().dtype()))) { return false; }
        return call.operation() != Operation.ROPE && call.operation() != Operation.ROPE_BACKWARD || call.attributes().number("base") > 1;
    }

    private Logic logic() { if (logic == null) { logic = Logic.create(device); } return logic; }
    private Cast casts() { if (cast == null) { cast = Cast.create(device); } return cast; }
    private Gather gathers() { if (gather == null) { gather = Gather.create(device); } return gather; }
    private Reduce reductions() { if (reduce == null) { reduce = Reduce.create(device); } return reduce; }
    private static StridedView vector(KernelTensor tensor) { return StridedView.contiguous(tensor.layout().offset(), tensor.layout().shape().elements()); }
    private static StridedView indexedView(long offset, long outer, long extent, long inner) {
        return StridedView.of(offset, new long[]{outer, extent, inner}, new long[]{extent * inner, inner, 1});
    }
    private void selected(Operation operation, AccuracyContract contract) { backend.selected(operation, "dynamis-gpu-primitives (exact)", contract); }
    private static AccuracyContract tensorContract(KernelCall call) {
        return new AccuracyContract(call.operation().name(), call.inputs().stream().map(t -> t.buffer().dtype().name()).toList(),
                "none", call.output().buffer().dtype().name(), AccuracyContract.Order.FIXED, AccuracyContract.Bound.exact(),
                "NNJava CPU " + call.operation() + " tensor semantics");
    }

    void encode(GpuCommandEncoder encoder, KernelCall call) {
        try (var scratch = new GpuScratch(backend)) {
            switch (call.operation()) {
                case GATHER, SCATTER_ADD -> indexed(encoder, call, scratch);
                case REDUCE_MAX, ARGMAX, MAX_BACKWARD -> maximum(encoder, call, scratch);
                case CONCAT, SLICE_BACKWARD -> copyLayout(encoder, call);
                case ROPE, ROPE_BACKWARD -> rotary(encoder, call, scratch);
                case ALL -> all(encoder, call, scratch);
                default -> logical(encoder, call, scratch);
            }
        }
    }

    private void logical(GpuCommandEncoder e, KernelCall call, GpuScratch scratch) {
        var shape = call.output().layout().shape();
        var a = scratch.broadcast(e, call.inputs().getFirst(), shape); var out = call.output();
        var av = vector(a); var ov = vector(out); var op = call.operation();
        if (op == Operation.CAST) {
            var from = Cast.Type.valueOf(a.buffer().dtype().name()); var to = Cast.Type.valueOf(out.buffer().dtype().name());
            casts().cast(e, from, to, backend.storage(a), av, backend.storage(out), ov);
            selected(op, Cast.contract(from, to)); return;
        }
        if (op == Operation.NOT) { logic().not(e, backend.storage(a), av, backend.storage(out), ov); }
        else if (op == Operation.IS_FINITE) { logic().classify(e, Logic.Classify.IS_FINITE, backend.storage(a), av, backend.storage(out), ov); }
        else {
            var b = scratch.broadcast(e, call.inputs().get(1), shape); var bv = vector(b);
            if (op == Operation.WHERE) {
                var c = scratch.broadcast(e, call.inputs().get(2), shape);
                logic().select(e, backend.storage(a), av, backend.storage(b), bv, backend.storage(c), vector(c), backend.storage(out), ov);
            } else if (op == Operation.AND || op == Operation.OR) {
                logic().logical(e, Logic.Logical.valueOf(op.name()), backend.storage(a), av, backend.storage(b), bv, backend.storage(out), ov);
            } else {
                var type = a.buffer().dtype() == DType.FP32 ? Logic.Operand.FP32 : Logic.Operand.INT32;
                logic().compare(e, Logic.Compare.valueOf(op.name()), type, backend.storage(a), av, backend.storage(b), bv, backend.storage(out), ov);
            }
        }
        selected(op, Logic.contract());
    }

    private void all(GpuCommandEncoder e, KernelCall call, GpuScratch scratch) {
        var input = scratch.packed(e, call.inputs().getFirst(), false); long n = input.layout().shape().elements();
        if (n == 0) { backend.fillWord(e, call.output(), 1); }
        else {
            var values = scratch.tensor(Shape.of(n)); var minimum = scratch.tensor(Shape.of());
            casts().cast(e, Cast.Type.BOOL, Cast.Type.FP32, backend.storage(input), vector(input), backend.storage(values), vector(values));
            reductions().rows(e, Reduce.Op.MIN, backend.storage(values), StridedView.matrix(1, n), backend.storage(minimum), vector(minimum));
            casts().cast(e, Cast.Type.FP32, Cast.Type.BOOL, backend.storage(minimum), vector(minimum), backend.storage(call.output()), vector(call.output()));
        }
        selected(call.operation(), tensorContract(call));
    }

    /** Validate even an empty gather domain. A stride-zero word represents the whole logical source axis. */
    private void validateIndices(GpuCommandEncoder e, KernelTensor indices, long extent, GpuScratch scratch) {
        long n = indices.layout().shape().elements();
        if (n == 0) { return; }
        var zero = scratch.tensor(DType.INT32, Shape.of(1)); backend.fillWord(e, zero, 0);
        var checked = scratch.tensor(DType.INT32, Shape.of(n));
        gathers().gather(e, backend.storage(zero), StridedView.of(0, new long[]{extent}, new long[]{0}), 0,
                backend.storage(indices), vector(indices), backend.storage(checked), vector(checked));
    }

    private void validatePositions(GpuCommandEncoder e, KernelTensor positions, GpuScratch scratch) {
        long n = positions.layout().shape().elements();
        var errors = scratch.tensor(DType.INT32, Shape.of(4)); backend.fillWord(e, errors, 0);
        var negative = scratch.tensor(DType.BOOL, Shape.of(n)); var flags = scratch.tensor(Shape.of(n)); var any = scratch.tensor(Shape.of());
        var firstWord = StridedView.contiguous(0, 1);
        logic().compare(e, Logic.Compare.LESS, Logic.Operand.INT32, backend.storage(positions), vector(positions),
                backend.storage(errors), firstWord, backend.storage(negative), vector(negative));
        casts().cast(e, Cast.Type.BOOL, Cast.Type.FP32, backend.storage(negative), vector(negative), backend.storage(flags), vector(flags));
        reductions().rows(e, Reduce.Op.MAX, backend.storage(flags), StridedView.matrix(1, n), backend.storage(any), vector(any));
        // KernelErrors: code bit 1 and one report when any position is negative; no individual index is retained.
        casts().cast(e, Cast.Type.FP32, Cast.Type.INT32, backend.storage(any), vector(any), backend.storage(errors), firstWord);
        try (var code = backend.storage(errors).slice(0, 4); var count = backend.storage(errors).slice(8, 4)) { e.copy(code, count); }
        e.checkErrors(backend.storage(errors));
    }

    private void indexed(GpuCommandEncoder e, KernelCall call, GpuScratch scratch) {
        boolean scatter = call.operation() == Operation.SCATTER_ADD;
        var source = scatter ? call.output().layout().shape() : call.inputs().getFirst().layout().shape();
        int axis = source.axis(call.attributes().intValue("axis"));
        long outer = source.prefix(axis).elements(), extent = source.dimension(axis), inner = 1;
        for (int i = axis + 1; i < source.rank(); i++) { inner *= source.dimension(i); }
        var indices = scratch.packed(e, call.inputs().get(1), false); long n = indices.layout().shape().elements();
        if (call.output().layout().shape().elements() == 0) { validateIndices(e, indices, extent, scratch); }
        else {
            var input = scratch.packed(e, call.inputs().getFirst(), false); var output = call.output();
            var original = indexedView(scatter ? output.layout().offset() : input.layout().offset(), outer, extent, inner);
            var expanded = indexedView(scatter ? input.layout().offset() : output.layout().offset(), outer, n, inner);
            if (scatter) { gathers().scatterAdd(e, backend.storage(input), expanded, 1, backend.storage(indices), vector(indices), backend.storage(output), original, false); }
            else { gathers().gather(e, backend.storage(input), original, 1, backend.storage(indices), vector(indices), backend.storage(output), expanded); }
        }
        selected(call.operation(), scatter ? Gather.scatterAddContract() : Gather.gatherContract());
    }

    private void copyLayout(GpuCommandEncoder e, KernelCall call) {
        var output = call.output(); var attributes = call.attributes(); int axis = output.layout().shape().axis(attributes.intValue("axis"));
        if (call.operation() == Operation.CONCAT) {
            long start = 0;
            for (var input : call.inputs()) {
                long end = start + input.layout().shape().dimension(axis);
                backend.copy(e, input, new KernelTensor(output.buffer(), output.layout().slice(axis, start, end, 1))); start = end;
            }
        } else {
            backend.fillWord(e, output, 0);
            var slice = output.layout().slice(axis, attributes.integer("start"), attributes.integer("end"), attributes.integer("step"));
            backend.copy(e, call.inputs().getFirst(), new KernelTensor(output.buffer(), slice));
        }
        backend.selected(call.operation(), "DynamisGPU storage", tensorContract(call));
    }

    private void maximum(GpuCommandEncoder e, KernelCall call, GpuScratch scratch) {
        var input = call.inputs().getFirst(); var shape = input.layout().shape(); boolean[] axes = new boolean[shape.rank()];
        for (int axis : call.attributes().axes()) { axes[shape.axis(axis)] = true; }
        int[] permutation = new int[shape.rank()], inverse = new int[shape.rank()]; int at = 0; long rows = 1, width = 1;
        for (int i = 0; i < axes.length; i++) { if (!axes[i]) { permutation[at++] = i; rows *= shape.dimension(i); } }
        for (int i = 0; i < axes.length; i++) { if (axes[i]) { permutation[at++] = i; width *= shape.dimension(i); } }
        for (int i = 0; i < inverse.length; i++) { inverse[permutation[i]] = i; }
        var permuted = new KernelTensor(input.buffer(), input.layout().permute(permutation));
        var packed = scratch.packed(e, permuted, true);
        var indices = call.operation() == Operation.ARGMAX ? call.output() : scratch.tensor(DType.INT32, Shape.of(rows));
        reductions().argmax(e, backend.storage(packed), StridedView.matrix(rows, width), backend.storage(indices), vector(indices));
        if (call.operation() == Operation.REDUCE_MAX) {
            // CPU max returns its first winning value, including the sign of a zero tie; Math.max differs.
            var offsets = scratch.tensor(DType.INT32, Shape.of(rows)); var absolute = scratch.tensor(DType.INT32, Shape.of(rows));
            var bytes = ByteBuffer.allocate(Math.toIntExact(rows * 4)).order(ByteOrder.LITTLE_ENDIAN);
            for (long row = 0; row < rows; row++) { bytes.putInt(Math.toIntExact(row * width)); }
            e.upload(backend.storage(offsets), bytes.array());
            logic().integer(e, Logic.IntOp.ADD, backend.storage(indices), vector(indices), backend.storage(offsets), vector(offsets), backend.storage(absolute), vector(absolute));
            gathers().gather(e, backend.storage(packed), StridedView.contiguous(shape.elements()), 0, backend.storage(absolute), vector(absolute), backend.storage(call.output()), vector(call.output()));
        } else if (call.operation() == Operation.MAX_BACKWARD) {
            var dy = scratch.packed(e, call.inputs().get(1), false); var dx = scratch.tensor(permuted.layout().shape());
            reductions().maxBackward(e, backend.storage(indices), vector(indices), backend.storage(dy), vector(dy), 1,
                    backend.storage(dx), StridedView.matrix(rows, width));
            backend.copy(e, new KernelTensor(dx.buffer(), dx.layout().permute(inverse)), call.output());
        }
        selected(call.operation(), tensorContract(call));
    }

    private void rotary(GpuCommandEncoder e, KernelCall call, GpuScratch scratch) {
        var input = call.inputs().getFirst(); var shape = input.layout().shape(); int dim = Math.toIntExact(shape.dimension(shape.rank() - 1));
        var rows = shape.prefix(shape.rank() - 1); int count = Math.toIntExact(rows.elements());
        var attributes = call.attributes(); boolean half = attributes.flag("half");
        var plan = Rope.plan(dim, attributes.number("base"), half ? Rope.Layout.HALF : Rope.Layout.INTERLEAVED);
        var frequencies = scratch.tensor(DType.INT32, Shape.of(dim));
        e.upload(backend.storage(frequencies), GpuRope.frequencies(plan.inverseFrequencies()));
        var positions = scratch.tensor(DType.INT32, Shape.of(count, 2)); backend.fillWord(e, positions, 0);
        if (call.inputs().size() == 2) {
            var ids = scratch.broadcast(e, call.inputs().get(1), rows);
            validatePositions(e, ids, scratch);
            casts().cast(e, Cast.Type.INT32, Cast.Type.INT32, backend.storage(ids), vector(ids), backend.storage(positions),
                    StridedView.of(0, new long[]{count}, new long[]{2}));
        } else {
            if (rows.dimension(rows.rank() - 1) != 1 && e instanceof GpuDecodeCapture) {
                throw new UnsupportedOperationException("Recorded rotary execution requires one-token decoding");
            }
            var bytes = ByteBuffer.allocate(Math.multiplyExact(count, 8)).order(ByteOrder.LITTLE_ENDIAN);
            long sequence = rows.dimension(rows.rank() - 1), offset = attributes.integer("offset");
            for (int row = 0; row < count; row++) { bytes.putLong(offset + row % sequence); }
            backend.rotaryPositions(e, backend.storage(positions), bytes.array(), count);
        }
        backend.copy(e, input, call.output());
        if (rope == null) { rope = new GpuRope(device); }
        var layout = half ? GpuRope.Layout.HALF : GpuRope.Layout.INTERLEAVED;
        if (call.operation() == Operation.ROPE) {
            rope.apply(e, dim, layout, backend.storage(frequencies), backend.storage(call.output()), call.output().layout().offset(), dim, count, backend.storage(positions), 0, 1);
        } else {
            rope.applyBackward(e, dim, layout, backend.storage(frequencies), backend.storage(call.output()), call.output().layout().offset(), dim, count, backend.storage(positions), 0, 1);
        }
        backend.selected(call.operation(), "vectrix-gpu", GpuNeuralKernels.exactContract(call.operation()));
    }

    @Override public void close() throws Exception {
        Exception failure = null;
        for (var resource : new AutoCloseable[]{logic, cast, gather, reduce, rope}) {
            if (resource != null) { try { resource.close(); } catch (Exception error) { if (failure == null) { failure = error; } else { failure.addSuppressed(error); } } }
        }
        if (failure != null) { throw failure; }
    }
}

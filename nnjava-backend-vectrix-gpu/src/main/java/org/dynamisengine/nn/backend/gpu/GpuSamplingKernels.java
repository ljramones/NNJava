package org.dynamisengine.nn.backend.gpu;

import java.util.List;
import org.dynamisengine.gpu.api.compute.GpuCapabilities;
import org.dynamisengine.gpu.api.compute.GpuCommandEncoder;
import org.dynamisengine.gpu.api.compute.GpuDevice;
import org.dynamisengine.gpu.api.compute.kit.AccuracyContract;
import org.dynamisengine.gpu.api.compute.kit.StridedView;
import org.dynamisengine.gpu.primitives.Cast;
import org.dynamisengine.gpu.primitives.Logic;
import org.dynamisengine.gpu.primitives.Reduce;
import org.dynamisengine.gpu.primitives.Sampler;
import org.dynamisengine.nn.backend.KernelCall;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.ops.SamplingOptions;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.vectrix.compute.GpuSampler;

/** Strict logit validation and optional provider sampling, without numerical readbacks or new shaders. */
final class GpuSamplingKernels implements AutoCloseable {
    private final GpuBackend backend;
    private final GpuDevice device;
    private Reduce reduce;
    private Logic logic;
    private Cast cast;
    private GpuSampler greedy;
    private Sampler sampler;

    GpuSamplingKernels(GpuBackend backend, GpuDevice device) { this.backend = backend; this.device = device; }
    static SamplingOptions options(KernelCall call) {
        var a = call.attributes(); return new SamplingOptions((float) a.number("temperature"), a.intValue("topK"), (float) a.number("topP"), a.integer("seed"));
    }
    static boolean supports(GpuCapabilities capabilities, SamplingOptions options, long rows, long vocabulary) {
        if (rows < 0 || rows > 65535 || vocabulary < 1 || vocabulary > Sampler.MAX_VOCAB || options.topK() > Sampler.MAX_K) { return false; }
        if (options.greedySelection()) { return true; }
        // The primitive's implicit top-p candidate cap is not a full-vocabulary nucleus. Require an explicit top-k.
        return (options.topP() == 1 || options.topK() > 0) && Float.isFinite(1f / options.temperature())
                && capabilities.maxGroupThreads() >= 1024 && capabilities.maxGroupSize().x() >= 1024;
    }
    void encode(GpuCommandEncoder e, KernelCall call) {
        try (var scratch = new GpuScratch(backend)) {
            var x = scratch.packed(e, call.inputs().getFirst(), true); var shape = x.layout().shape();
            int width = Math.toIntExact(shape.dimension(shape.rank() - 1)), rows = Math.toIntExact(shape.elements() / width);
            if (reduce == null) { reduce = Reduce.create(device); } if (logic == null) { logic = Logic.create(device); } if (cast == null) { cast = Cast.create(device); }
            var maxima = scratch.tensor(Shape.of(rows)); var finite = scratch.tensor(DType.BOOL, Shape.of(rows));
            var invalid = scratch.tensor(DType.BOOL, Shape.of(rows)); var flags = scratch.tensor(Shape.of(rows));
            var any = scratch.tensor(Shape.of()); var errors = scratch.tensor(DType.INT32, Shape.of(4)); backend.fillWord(e, errors, 0);
            reduce.rows(e, Reduce.Op.MAX, backend.storage(x), StridedView.matrix(rows, width), backend.storage(maxima), StridedView.contiguous(rows));
            // Max is finite iff the row has no NaN/+infinity and at least one finite candidate.
            logic.classify(e, Logic.Classify.IS_FINITE, backend.storage(maxima), StridedView.contiguous(rows), backend.storage(finite), StridedView.contiguous(rows));
            logic.not(e, backend.storage(finite), StridedView.contiguous(rows), backend.storage(invalid), StridedView.contiguous(rows));
            cast.cast(e, Cast.Type.BOOL, Cast.Type.FP32, backend.storage(invalid), StridedView.contiguous(rows), backend.storage(flags), StridedView.contiguous(rows));
            reduce.rows(e, Reduce.Op.MAX, backend.storage(flags), StridedView.matrix(1, rows), backend.storage(any), StridedView.contiguous(1));
            cast.cast(e, Cast.Type.FP32, Cast.Type.INT32, backend.storage(any), StridedView.contiguous(1), backend.storage(errors), StridedView.contiguous(1));
            try (var code = backend.storage(errors).slice(0, 4); var count = backend.storage(errors).slice(8, 4)) { e.copy(code, count); }
            e.checkErrors(backend.storage(errors));
            var options = options(call); var output = call.output();
            if (options.greedySelection()) {
                if (greedy == null) { greedy = new GpuSampler(device); }
                greedy.argmax(e, rows, backend.storage(x), 0, width, width, backend.storage(output), output.layout().offset());
                backend.selected(Operation.SAMPLE, "vectrix-gpu", new AccuracyContract("SAMPLE greedy", List.of("FP32 logits"), "none", "INT32 token",
                        AccuracyContract.Order.FIXED, AccuracyContract.Bound.exact(), "Vectrix Sampler.argmax on valid rows"));
            } else {
                if (sampler == null) { sampler = Sampler.create(device); }
                // The primitive writes at zero; runtime copies handle offset output views without host reads.
                var target = output.layout().offset() == 0 ? output : scratch.tensor(DType.INT32, output.layout().shape());
                sampler.encode(e, backend.storage(x), backend.storage(target), Sampler.Shape.of(rows, width),
                        new Sampler.Settings(options.temperature(), options.topK(), options.topP(), options.seed()), (int) call.attributes().integer("step"));
                if (target != output) { backend.copy(e, target, output); }
                backend.selected(Operation.SAMPLE, "dynamis-gpu-primitives", Sampler.contract());
            }
        }
    }
    @Override public void close() throws Exception {
        Exception failure = null;
        for (var resource : new AutoCloseable[]{sampler, greedy, cast, logic, reduce}) {
            if (resource != null) { try { resource.close(); } catch (Exception error) { if (failure == null) { failure = error; } else { failure.addSuppressed(error); } } }
        }
        if (failure != null) { throw failure; }
    }
}

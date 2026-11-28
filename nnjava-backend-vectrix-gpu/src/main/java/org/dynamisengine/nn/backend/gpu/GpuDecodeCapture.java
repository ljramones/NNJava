package org.dynamisengine.nn.backend.gpu;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.primitives.GemmF16;
import org.dynamisengine.nn.backend.CommandCapture;
import org.dynamisengine.nn.backend.CommandRecording;
import org.dynamisengine.nn.backend.Completion;

/** Collects the entire fixed decoder into one upstream recording; error readbacks run separately. */
final class GpuDecodeCapture implements CommandCapture, GpuCommandEncoder {
    private record PositionUpload(GpuBuffer source, int rows) { }
    private record WritePatch(int dispatch, byte[] parameters) { }
    private final GpuBackend backend;
    private final GpuDevice device;
    private final GpuCommandEncoder encoder;
    private final List<GpuBuffer> errors = new ArrayList<>();
    private final List<PositionUpload> positions = new ArrayList<>();
    private final List<WritePatch> writes = new ArrayList<>();
    private int dispatches;
    private int positionLimit = Integer.MAX_VALUE;
    private boolean finished, closed;

    GpuDecodeCapture(GpuBackend backend, GpuDevice device) {
        this.backend = backend; this.device = device; encoder = device.encoder().name("nnjava.decode");
    }
    void limitPosition(int limit) { positionLimit = Math.min(positionLimit, limit); }
    void positions(GpuBuffer target, int rows) {
        if (rows <= 0) { throw new IllegalArgumentException("Empty rotary positions"); }
        var source = device.allocate(new BufferDescriptor(Math.multiplyExact(rows, 8L),
                Set.of(BufferUsage.MAP_WRITE, BufferUsage.TRANSFER_SOURCE), Placement.SHARED, "nnjava-replay-position"));
        positions.add(new PositionUpload(source, rows)); encoder.copy(source, target);
    }
    @Override public GpuCommandEncoder upload(GpuBuffer target, byte[] data) { encoder.upload(target, data); return this; }
    @Override public GpuReadback download(GpuBuffer source) { throw new UnsupportedOperationException("Readbacks cannot be captured"); }
    @Override public GpuCommandEncoder copy(GpuBuffer source, GpuBuffer target) { encoder.copy(source, target); return this; }
    @Override public GpuCommandEncoder fill(GpuBuffer target, int pattern) { encoder.fill(target, pattern); return this; }
    @Override public GpuCommandEncoder resetErrors(GpuBuffer buffer) { encoder.resetErrors(buffer); return this; }
    @Override public GpuCommandEncoder checkErrors(GpuBuffer buffer) {
        // A primitive can reset/reuse its report buffer later in this graph. Snapshot each check at its original point.
        var snapshot = device.allocate(KernelErrors.descriptor().withLabel("nnjava-replay-error"));
        errors.add(snapshot);
        try (var report = buffer.slice(0, KernelErrors.BYTES)) { encoder.copy(report, snapshot); }
        return this;
    }
    @Override public GpuCommandEncoder label(String label) { encoder.label(label); return this; }
    @Override public GpuCommandEncoder dispatch(GpuComputePipeline pipeline, Map<Integer, ? extends GpuBuffer> bindings, byte[] parameters, Dimensions groups) {
        return dispatch(pipeline, bindings, parameters, groups, "");
    }
    @Override public GpuCommandEncoder dispatch(GpuComputePipeline pipeline, Map<Integer, ? extends GpuBuffer> bindings,
            byte[] parameters, Dimensions groups, String label) {
        if ("kv.write".equals(label)) { writes.add(new WritePatch(dispatches, parameters.clone())); }
        encoder.dispatch(pipeline, bindings, parameters, groups, label); dispatches++; return this;
    }
    @Override public CommandRecording finish() {
        if (closed || finished) { throw new IllegalStateException("Capture is no longer open"); }
        // Obtain the patch field from the published manifest, never a locally duplicated byte offset.
        int positionOffset = writes.isEmpty() ? -1 : KernelLibrary.load(GemmF16.class, "kernels", "kernels.properties", "dynamisgpu.primitives.")
                .parameterOffsets("kv_write").get("position");
        GpuRecording recording = encoder.record(); finished = true; backend.endCapture(this);
        var result = new CommandRecording() {
            private boolean released;
            public int dispatches() { return recording.dispatches(); }
            public Completion run(int position) {
                if (released) { throw new IllegalStateException("Recording is closed"); }
                if (position < 0 || position >= positionLimit) { throw new IllegalArgumentException("Position exceeds the recorded cache domain"); }
                for (var upload : positions) {
                    try (var mapping = upload.source().map(Access.WRITE)) {
                        var bytes = mapping.bytes().order(ByteOrder.LITTLE_ENDIAN);
                        for (int row = 0; row < upload.rows(); row++) { bytes.putLong(position); }
                    }
                }
                for (var patch : writes) {
                    byte[] parameters = patch.parameters().clone();
                    ByteBuffer.wrap(parameters).order(ByteOrder.LITTLE_ENDIAN).putInt(positionOffset, position);
                    recording.parameters(patch.dispatch(), parameters);
                }
                // Finish both phases before making logits visible or reporting a committed cache length.
                new GpuCompletion(device.queue().submit(recording), () -> { }).await();
                if (!errors.isEmpty()) {
                    try (var checks = device.encoder().name("nnjava.decode.errors")) {
                        errors.forEach(checks::checkErrors);
                        new GpuCompletion(device.queue().submit(checks), () -> { }).await();
                    }
                }
                return Completion.completed();
            }
            public void close() {
                if (!released) {
                    released = true;
                    try { recording.close(); } finally { releaseBuffers(); backend.releaseInferenceResource(this); }
                }
            }
        };
        backend.ownInferenceResource(result); return result;
    }
    private void releaseBuffers() {
        for (var upload : positions) { upload.source().close(); }
        errors.forEach(GpuBuffer::close); positions.clear(); errors.clear();
    }
    @Override public void close() {
        if (!closed) {
            closed = true;
            try { encoder.close(); } finally {
                backend.endCapture(this);
                if (!finished) { releaseBuffers(); }
            }
        }
    }
}

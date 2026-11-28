package org.dynamisengine.nn.nn;

import java.util.Map;
import java.util.Objects;
import java.util.Set;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.state.TensorSnapshot;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;

/** Owns retained hidden/cell handles. States are explicit values; modules never mutate them. */
public final class RecurrentState implements AutoCloseable {
    private final Tensor hidden, cell;
    private boolean closed;

    private RecurrentState(Tensor hidden, Tensor cell) {
        Objects.requireNonNull(hidden);
        if (hidden.dtype()!=DType.FP32 || hidden.shape().rank()!=2 || hidden.shape().dimension(1)<1) {
            throw new IllegalArgumentException("Recurrent state must be FP32 [batch,hidden]");
        }
        if (cell!=null) {
            hidden.context().requireContext(cell);
            if (cell.dtype()!=DType.FP32 || !cell.shape().equals(hidden.shape())) { throw new IllegalArgumentException("LSTM hidden/cell shapes and types must match"); }
        }
        this.hidden=hidden.retain();
        try { this.cell=cell==null ? null : cell.retain(); }
        catch (RuntimeException | Error error) { this.hidden.close(); throw error; }
    }

    /** Borrows the tensor and retains an independent owned handle. */
    public static RecurrentState hidden(Tensor hidden) { return new RecurrentState(hidden,null); }
    public static RecurrentState lstm(Tensor hidden, Tensor cell) { return new RecurrentState(hidden,Objects.requireNonNull(cell)); }
    public Tensor hidden() { ensureOpen(); return hidden.retain(); }
    public Tensor cell() {
        ensureOpen();
        if (cell==null) { throw new IllegalStateException("This recurrent state has no LSTM cell value"); }
        return cell.retain();
    }
    public boolean hasCell() { ensureOpen(); return cell!=null; }
    public Shape shape() { ensureOpen(); return hidden.shape(); }
    public RecurrentState retain() { ensureOpen(); return new RecurrentState(hidden,cell); }

    /** Shared values without gradient history, for crossing tape/truncated-BPTT boundaries. */
    public RecurrentState detach() {
        ensureOpen();
        try (var h=hidden.detach(); var c=cell==null ? null : cell.detach()) { return new RecurrentState(h,c); }
    }

    /** Returns a new state with selected rows zeroed; other rows retain their history. */
    public RecurrentState resetRows(Tensor reset) {
        ensureOpen(); validateMask(reset,hidden);
        try (var mask=Ops.reshape(reset,Shape.of(shape().dimension(0),1)); var zero=hidden.context().full(Shape.of(),0);
             var h=Ops.where(mask,zero,hidden); var c=cell==null ? null : Ops.where(mask,zero,cell)) { return new RecurrentState(h,c); }
    }

    /** Neutral host snapshots; applications persist carried state alongside stream position. */
    public Map<String,TensorSnapshot> snapshot() {
        ensureOpen();
        return cell==null ? Map.of("hidden",TensorSnapshot.capture(hidden)) : Map.of("hidden",TensorSnapshot.capture(hidden),"cell",TensorSnapshot.capture(cell));
    }

    /** Materializes detached values, independent of any original context or tape. */
    public static RecurrentState fromSnapshot(ExecutionContext context,Map<String,TensorSnapshot> snapshot) {
        var values=Map.copyOf(snapshot);
        if (!values.keySet().equals(Set.of("hidden")) && !values.keySet().equals(Set.of("hidden","cell"))) {
            throw new IllegalArgumentException("Recurrent snapshots require hidden and optional cell tensors");
        }
        var h=values.get("hidden"); var c=values.get("cell");
        if (h.dtype()!=DType.FP32 || h.shape().rank()!=2 || h.shape().dimension(1)<1
                || c!=null && (c.dtype()!=DType.FP32 || !c.shape().equals(h.shape()))) { throw new IllegalArgumentException("Invalid recurrent snapshot shapes/types"); }
        try (var hidden=h.materialize(context); var cell=c==null ? null : c.materialize(context)) { return new RecurrentState(hidden,cell); }
    }

    void validate(ExecutionContext context,long batch,long width,boolean lstm) {
        ensureOpen(); context.requireContext(hidden);
        if (!hidden.shape().equals(Shape.of(batch,width)) || (cell!=null)!=lstm) { throw new IllegalArgumentException("Recurrent state shape/kind mismatch"); }
    }

    static RecurrentState selectRows(Tensor active,RecurrentState yes,RecurrentState no) {
        yes.ensureOpen(); no.ensureOpen(); validateMask(active,yes.hidden);
        no.validate(yes.hidden.context(),yes.shape().dimension(0),yes.shape().dimension(1),yes.hasCell());
        try (var mask=Ops.reshape(active,Shape.of(yes.shape().dimension(0),1)); var h=Ops.where(mask,yes.hidden,no.hidden);
             var c=yes.cell==null ? null : Ops.where(mask,yes.cell,no.cell)) { return new RecurrentState(h,c); }
    }

    private static void validateMask(Tensor mask,Tensor hidden) {
        hidden.context().requireContext(mask);
        if (mask.dtype()!=DType.BOOL || !mask.shape().equals(Shape.of(hidden.shape().dimension(0)))) { throw new IllegalArgumentException("State row masks must be BOOL [batch]"); }
    }

    private void ensureOpen() {
        if (closed) { throw new IllegalStateException("Recurrent state is closed"); }
        hidden.context().ensureOpen();
    }
    @Override public void close() {
        // Tensor.close permits cleanup after context closure while enforcing the owner thread.
        if (!closed) { hidden.close(); if (cell!=null) { cell.close(); } closed=true; }
    }
}

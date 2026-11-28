package org.dynamisengine.nn.nn;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.model.ComponentSpec;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;

/** Stateless recurrent sequence unrolling over trailing [time,features] axes. */
public final class Recurrent extends AbstractModule implements UnaryModule {
    private final Definition definition;
    private final RecurrentCell cell;

    private Recurrent(ExecutionContext context,Definition definition) {
        super(context); this.definition=definition; definition.validate(context);
        try {
            cell=register("cell",new RecurrentCell.Definition("cell",definition.kind,definition.inputFeatures,definition.hiddenSize,
                    definition.bias,ModelBuild.seed(definition.seed,"cell")).build(context));
        } catch (RuntimeException | Error error) { close(); throw error; }
    }

    public static Builder builder() { return new Builder(); }
    public Definition definition() { ensureOpen(); return definition; }
    /** Borrowed cell, for step-by-step use of the same registered parameters. */
    public RecurrentCell cell() { ensureOpen(); return cell; }
    public RecurrentState zeroState(long batchSize) { ensureOpen(); return cell.zeroState(batchSize); }

    @Override public Tensor forward(Tensor input) {
        try (var result=run(input)) { return result.output(); }
    }

    public Result run(Tensor input) { return run(input,null,null); }
    public Result run(Tensor input,RecurrentState initial) { return run(input,initial,null); }

    /**
     * Borrows inputs/state/INT32 lengths. Padded outputs are zero; final states hold
     * at each row's valid length. No state is stored in the module between calls.
     */
    public Result run(Tensor input,RecurrentState initial,Tensor lengths) {
        ensureOpen(); context.requireContext(input);
        Shape outputShape=definition.outputShape(input.shape());
        if (input.dtype()!=DType.FP32) { throw new IllegalArgumentException("Recurrent input must be FP32"); }
        Shape prefix=input.shape().prefix(input.shape().rank()-2);
        long batch=prefix.elements(), time=input.shape().dimension(input.shape().rank()-2);
        if (batch<1) { throw new IllegalArgumentException("Recurrent sequences require a nonempty batch"); }
        if (initial!=null) { initial.validate(context,batch,definition.hiddenSize,definition.kind==RecurrentCell.Kind.LSTM); }
        if (lengths!=null) {
            context.requireContext(lengths);
            if (lengths.dtype()!=DType.INT32 || !lengths.shape().equals(prefix)) { throw new IllegalArgumentException("Lengths must be INT32 with the input batch-prefix shape"); }
            for (int length : lengths.toIntArray()) { if (length<0 || length>time) { throw new IllegalArgumentException("Sequence length is outside [0,time]"); } }
        }
        RecurrentState current=initial==null ? cell.zeroState(batch) : initial.retain();
        var outputs=new ArrayList<Tensor>();
        try (var packed=Ops.contiguous(input); var sequences=Ops.reshape(packed,Shape.of(batch,time,definition.inputFeatures));
             var lengthValues=lengths==null ? null : Ops.contiguous(lengths);
             var flatLengths=lengthValues==null ? null : Ops.reshape(lengthValues,Shape.of(batch));
             var zero=lengths==null ? null : context.full(Shape.of(),0)) {
            for (long step=0;step<time;step++) {
                try (var slice=Ops.slice(sequences,1,step,step+1); var row=Ops.contiguous(slice);
                     var matrix=Ops.reshape(row,Shape.of(batch,definition.inputFeatures));
                     var index=lengths==null ? null : context.tensor(new int[]{Math.toIntExact(step)},Shape.of());
                     var active=lengths==null ? null : Ops.less(index,flatLengths);
                     var mask=active==null ? null : Ops.reshape(active,Shape.of(batch,1));
                     var value=mask==null ? matrix.retain() : Ops.where(mask,matrix,zero);
                     var proposed=cell.step(value,current)) {
                    RecurrentState next=active==null ? proposed.retain() : RecurrentState.selectRows(active,proposed,current);
                    current.close(); current=next;
                    if (definition.output==SequenceOutput.ALL) {
                        try (var h=current.hidden(); var selected=mask==null ? h.retain() : Ops.where(mask,h,zero)) {
                            outputs.add(Ops.reshape(selected,Shape.of(batch,1,definition.hiddenSize)));
                        }
                    }
                }
            }
            Tensor output;
            if (definition.output==SequenceOutput.ALL) {
                try (var joined=Ops.concat(1,outputs.toArray(Tensor[]::new))) { output=Ops.reshape(joined,outputShape); }
            } else {
                try (var hidden=current.hidden(); var contiguous=Ops.contiguous(hidden)) { output=Ops.reshape(contiguous,outputShape); }
            }
            var result=new Result(output,current); current=null; return result;
        } finally {
            outputs.forEach(Tensor::close);
            if (current!=null) { current.close(); }
        }
    }

    @Override protected Map<String,String> localConfiguration() { return ModelBuild.configuration(definition.specification()); }

    static void validate(ExecutionContext context,String path,RecurrentCell.Kind kind,long inputs,long hidden,boolean bias) {
        RecurrentCell.validate(context,path+".cell",kind,inputs,hidden,bias);
        ModelBuild.require(context,path,DType.FP32,Operation.CONCAT,Operation.SLICE_BACKWARD,Operation.WHERE);
        ModelBuild.require(context,path,DType.BOOL,Operation.LESS);
    }

    public record Definition(String name,RecurrentCell.Kind kind,long inputFeatures,long hiddenSize,boolean bias,SequenceOutput output,long seed)
            implements ModelDefinition<Recurrent> {
        public Definition {
            new RecurrentCell.Definition(name,kind,inputFeatures,hiddenSize,bias,seed); Objects.requireNonNull(output);
        }
        public Shape outputShape(Shape input) {
            if (input.rank()<2 || input.dimension(input.rank()-2)<1 || input.dimension(input.rank()-1)!=inputFeatures) {
                throw new IllegalArgumentException(name+": expected trailing [positiveTime,"+inputFeatures+"] axes");
            }
            return output==SequenceOutput.ALL ? input.replace(input.rank()-1,hiddenSize) : input.prefix(input.rank()-2).append(hiddenSize);
        }
        @Override public void validate(ExecutionContext context) { Recurrent.validate(context,name,kind,inputFeatures,hiddenSize,bias); }
        @Override public Recurrent build(ExecutionContext context) { return new Recurrent(context,this); }
        @Override public ComponentSpec specification() {
            return new ComponentSpec("nnjava.recurrent",1,name,Map.of("kind",kind.name(),"inputs",Long.toString(inputFeatures),"hidden",Long.toString(hiddenSize),
                    "bias",Boolean.toString(bias),"output",output.name(),"seed",Long.toString(seed)),List.of());
        }
        public static Definition fromSpec(ComponentSpec spec) {
            spec.require("nnjava.recurrent",1,"kind","inputs","hidden","bias","output","seed");
            if (!spec.children().isEmpty()) { throw new IllegalArgumentException("Recurrent definition cannot have children"); }
            return new Definition(spec.name(),RecurrentCell.Kind.valueOf(spec.attribute("kind")),spec.longValue("inputs"),spec.longValue("hidden"),
                    spec.booleanValue("bias"),SequenceOutput.valueOf(spec.attribute("output")),spec.longValue("seed"));
        }
    }

    /** Owns its output and final state; getters return additional owned handles. */
    public static final class Result implements AutoCloseable {
        private final Tensor output;
        private final RecurrentState state;
        private boolean closed;
        private Result(Tensor output,RecurrentState state) { this.output=output; this.state=state; }
        public Tensor output() { ensureOpen(); return output.retain(); }
        public RecurrentState state() { ensureOpen(); return state.retain(); }
        private void ensureOpen() { if (closed) { throw new IllegalStateException("Recurrent result is closed"); } output.context().ensureOpen(); }
        @Override public void close() { if (!closed) { output.close(); state.close(); closed=true; } }
    }

    public static final class Builder {
        private String name="recurrent";
        private RecurrentCell.Kind kind=RecurrentCell.Kind.RNN;
        private long inputs,hidden,seed;
        private boolean bias=true;
        private SequenceOutput output=SequenceOutput.ALL;
        private Builder() { }
        public Builder name(String value) { ModelBuild.name(value); name=value; return this; }
        public Builder kind(RecurrentCell.Kind value) { kind=Objects.requireNonNull(value); return this; }
        public Builder inputFeatures(long value) { inputs=value; return this; }
        public Builder hiddenSize(long value) { hidden=value; return this; }
        public Builder bias(boolean value) { bias=value; return this; }
        public Builder output(SequenceOutput value) { output=Objects.requireNonNull(value); return this; }
        public Builder seed(long value) { seed=value; return this; }
        public Definition define() { return new Definition(name,kind,inputs,hidden,bias,output,seed); }
        public Recurrent build(ExecutionContext context) { return define().build(context); }
    }
}

package org.dynamisengine.nn.nn;

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

/** RNN(tanh), LSTM(IFGO), or reset-after GRU(RZN), composed from provider-backed Ops. */
public final class RecurrentCell extends AbstractModule {
    public enum Kind {
        RNN(1), LSTM(4), GRU(3);
        private final int gates;
        Kind(int gates) { this.gates=gates; }
        public int gates() { return gates; }
    }
    private final Definition definition;
    private final Linear input, recurrent;

    private RecurrentCell(ExecutionContext context,Definition definition) {
        super(context); this.definition=definition; definition.validate(context);
        long width=Math.multiplyExact(definition.hiddenSize,definition.kind.gates);
        try {
            input=register("input",new Linear(context,definition.inputFeatures,width,ModelBuild.seed(definition.seed,"input"),definition.bias));
            recurrent=register("recurrent",new Linear(context,definition.hiddenSize,width,ModelBuild.seed(definition.seed,"recurrent"),definition.bias));
        } catch (RuntimeException | Error error) { close(); throw error; }
    }

    public static Builder builder() { return new Builder(); }
    public Definition definition() { ensureOpen(); return definition; }

    public RecurrentState zeroState(long batchSize) {
        ensureOpen();
        if (batchSize<0) { throw new IllegalArgumentException("Negative recurrent batch size"); }
        try (var h=context.zeros(DType.FP32,Shape.of(batchSize,definition.hiddenSize));
             var c=definition.kind==Kind.LSTM ? context.zeros(DType.FP32,h.shape()) : null) {
            return c==null ? RecurrentState.hidden(h) : RecurrentState.lstm(h,c);
        }
    }

    /** Borrows [batch,inputs] and state; returns a separately owned next state. */
    public RecurrentState step(Tensor value,RecurrentState state) {
        ensureOpen(); context.requireContext(value);
        if (value.dtype()!=DType.FP32 || value.shape().rank()!=2 || value.shape().dimension(1)!=definition.inputFeatures) {
            throw new IllegalArgumentException("Cell input must be FP32 [batch,"+definition.inputFeatures+"]");
        }
        Objects.requireNonNull(state).validate(context,value.shape().dimension(0),definition.hiddenSize,definition.kind==Kind.LSTM);
        try (var previous=state.hidden(); var x=input.forward(value); var h=recurrent.forward(previous)) {
            return switch (definition.kind) {
                case RNN -> rnn(x,h);
                case LSTM -> lstm(x,h,state);
                case GRU -> gru(x,h,previous);
            };
        }
    }

    private RecurrentState rnn(Tensor x,Tensor h) {
        try (var sum=Ops.add(x,h); var value=Ops.tanh(sum)) { return RecurrentState.hidden(value); }
    }

    private RecurrentState lstm(Tensor x,Tensor h,RecurrentState previous) {
        try (var gates=Ops.add(x,h); var i0=gate(gates,0); var f0=gate(gates,1); var g0=gate(gates,2); var o0=gate(gates,3);
             var i=Ops.sigmoid(i0); var f=Ops.sigmoid(f0); var g=Ops.tanh(g0); var o=Ops.sigmoid(o0); var oldCell=previous.cell();
             var carry=Ops.multiply(f,oldCell); var update=Ops.multiply(i,g); var cell=Ops.add(carry,update);
             var activated=Ops.tanh(cell); var hidden=Ops.multiply(o,activated)) { return RecurrentState.lstm(hidden,cell); }
    }

    private RecurrentState gru(Tensor x,Tensor h,Tensor previous) {
        try (var xr=gate(x,0); var xz=gate(x,1); var xn=gate(x,2); var hr=gate(h,0); var hz=gate(h,1); var hn=gate(h,2);
             var rs=Ops.add(xr,hr); var zs=Ops.add(xz,hz); var r=Ops.sigmoid(rs); var z=Ops.sigmoid(zs);
             var reset=Ops.multiply(r,hn); var sum=Ops.add(xn,reset); var candidate=Ops.tanh(sum);
             var difference=Ops.subtract(previous,candidate); var carry=Ops.multiply(z,difference); var hidden=Ops.add(candidate,carry)) {
            return RecurrentState.hidden(hidden);
        }
    }

    private Tensor gate(Tensor tensor,int index) {
        long start=Math.multiplyExact(definition.hiddenSize,index);
        return Ops.slice(tensor,1,start,Math.addExact(start,definition.hiddenSize));
    }

    @Override protected Map<String,String> localConfiguration() { return ModelBuild.configuration(definition.specification()); }

    static void validate(ExecutionContext context,String path,Kind kind,long inputs,long hidden,boolean bias) {
        ModelBuild.context(context);
        long width=Math.multiplyExact(hidden,Objects.requireNonNull(kind).gates);
        if (inputs<1 || hidden<1) { throw new IllegalArgumentException("Recurrent input/hidden sizes must be positive"); }
        ModelBuild.parameter(context,path+".input.weight",Shape.of(width,inputs),true);
        ModelBuild.parameter(context,path+".recurrent.weight",Shape.of(width,hidden),true);
        if (bias) {
            ModelBuild.parameter(context,path+".input.bias",Shape.of(width),false);
            ModelBuild.parameter(context,path+".recurrent.bias",Shape.of(width),false);
        }
        ModelBuild.require(context,path,DType.FP32,Operation.COPY,Operation.FILL,Operation.MATMUL,Operation.ADD,Operation.SUM_TO,Operation.TANH,Operation.TANH_BACKWARD);
        if (kind!=Kind.RNN) {
            ModelBuild.require(context,path,DType.FP32,Operation.SLICE_BACKWARD,Operation.MULTIPLY,Operation.SIGMOID,Operation.SIGMOID_BACKWARD);
        }
        if (kind==Kind.GRU) { ModelBuild.require(context,path,DType.FP32,Operation.SUBTRACT); }
    }

    public record Definition(String name,Kind kind,long inputFeatures,long hiddenSize,boolean bias,long seed) implements ModelDefinition<RecurrentCell> {
        public Definition {
            ModelBuild.name(name); Objects.requireNonNull(kind);
            if (inputFeatures<1 || hiddenSize<1) { throw new IllegalArgumentException("Recurrent input/hidden sizes must be positive"); }
            long width=Math.multiplyExact(hiddenSize,kind.gates); Shape.of(width,inputFeatures); Shape.of(width,hiddenSize);
        }
        @Override public void validate(ExecutionContext context) { RecurrentCell.validate(context,name,kind,inputFeatures,hiddenSize,bias); }
        @Override public RecurrentCell build(ExecutionContext context) { return new RecurrentCell(context,this); }
        @Override public ComponentSpec specification() {
            return new ComponentSpec("nnjava.recurrent-cell",1,name,Map.of("kind",kind.name(),"inputs",Long.toString(inputFeatures),
                    "hidden",Long.toString(hiddenSize),"bias",Boolean.toString(bias),"seed",Long.toString(seed)),List.of());
        }
        public static Definition fromSpec(ComponentSpec spec) {
            spec.require("nnjava.recurrent-cell",1,"kind","inputs","hidden","bias","seed");
            if (!spec.children().isEmpty()) { throw new IllegalArgumentException("Cell definition cannot have children"); }
            return new Definition(spec.name(),Kind.valueOf(spec.attribute("kind")),spec.longValue("inputs"),spec.longValue("hidden"),spec.booleanValue("bias"),spec.longValue("seed"));
        }
    }

    public static final class Builder {
        private String name="cell";
        private Kind kind=Kind.RNN;
        private long inputs,hidden,seed;
        private boolean bias=true;
        private Builder() { }
        public Builder name(String value) { ModelBuild.name(value); name=value; return this; }
        public Builder kind(Kind value) { kind=Objects.requireNonNull(value); return this; }
        public Builder inputFeatures(long value) { inputs=value; return this; }
        public Builder hiddenSize(long value) { hidden=value; return this; }
        public Builder bias(boolean value) { bias=value; return this; }
        public Builder seed(long value) { seed=value; return this; }
        public Definition define() { return new Definition(name,kind,inputs,hidden,bias,seed); }
        public RecurrentCell build(ExecutionContext context) { return define().build(context); }
    }
}

package org.dynamisengine.nn.nn;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Stream;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.model.ComponentSpec;
import org.dynamisengine.nn.ops.Conv2dConfig;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;

/** Trainable channel-first cross-correlation; weights are [outputs,inputs/groups,kernelH,kernelW]. */
public final class Conv2d extends AbstractModule implements UnaryModule {
    private final Definition definition;
    private final Parameter weight, bias;

    public Conv2d(ExecutionContext context, long inputs, long outputs, Conv2dConfig config, long seed) {
        this(context,new Definition("conv2d",inputs,outputs,config,true,seed));
    }

    public Conv2d(ExecutionContext context, long inputs, long outputs, Conv2dConfig config, long seed, boolean bias) {
        this(context,new Definition("conv2d",inputs,outputs,config,bias,seed));
    }

    private Conv2d(ExecutionContext context, Definition definition) {
        super(context);
        this.definition=definition;
        definition.validate(context);
        try {
            Shape shape=definition.config.weightShape(definition.inputChannels,definition.outputChannels);
            double fanIn=(double)(definition.inputChannels/definition.config.groups())*definition.config.kernelHeight()*definition.config.kernelWidth();
            try (var initial=context.normal(shape,definition.seed,(float)Math.sqrt(1/fanIn),true)) { weight=register("weight",initial); }
            bias=definition.bias ? constantParameter("bias",Shape.of(definition.outputChannels),0) : null;
        } catch (RuntimeException | Error error) { close(); throw error; }
    }

    public static Builder builder() { return new Builder(); }
    public Definition definition() { ensureOpen(); return definition; }

    @Override public Tensor forward(Tensor input) {
        ensureOpen(); context.requireContext(input); definition.outputShape(input.shape());
        try (var weights=weight.value(); var biases=bias==null ? null : bias.value()) {
            return Ops.conv2d(input,weights,biases,definition.config);
        }
    }

    @Override protected Map<String,String> localConfiguration() { return ModelBuild.configuration(definition.specification()); }

    public record Definition(String name, long inputChannels, long outputChannels, Conv2dConfig config, boolean bias, long seed)
            implements ModelDefinition<Conv2d> {
        public Definition { ModelBuild.name(name); Objects.requireNonNull(config).weightShape(inputChannels,outputChannels); }
        public Shape outputShape(Shape input) {
            if (Conv2dConfig.imageShape(input).dimension(0)!=inputChannels) { throw new IllegalArgumentException(name+": expected "+inputChannels+" input channels"); }
            return config.outputShape(input,outputChannels);
        }
        @Override public void validate(ExecutionContext context) {
            Conv2d.validate(context,inputChannels,outputChannels,config,bias,name);
        }
        @Override public Conv2d build(ExecutionContext context) { return new Conv2d(context,this); }
        @Override public ComponentSpec specification() {
            var attributes=new LinkedHashMap<>(settings(config));
            attributes.put("inputs",Long.toString(inputChannels)); attributes.put("outputs",Long.toString(outputChannels));
            attributes.put("bias",Boolean.toString(bias)); attributes.put("seed",Long.toString(seed));
            return new ComponentSpec("nnjava.conv2d",1,name,attributes,List.of());
        }
        public static Definition fromSpec(ComponentSpec spec) {
            require(spec,"nnjava.conv2d","inputs","outputs","bias","seed");
            return new Definition(spec.name(),spec.longValue("inputs"),spec.longValue("outputs"),settings(spec),spec.booleanValue("bias"),spec.longValue("seed"));
        }
    }

    static void validate(ExecutionContext context,long inputs,long outputs,Conv2dConfig config,boolean bias,String path) {
        ModelBuild.context(context);
        ModelBuild.parameter(context,path+".weight",config.weightShape(inputs,outputs),true);
        if (bias) { ModelBuild.parameter(context,path+".bias",Shape.of(outputs),false); }
        ModelBuild.require(context,path,DType.FP32,Operation.COPY,Operation.FILL,Operation.CONV2D,Operation.CONV2D_BACKWARD);
    }

    static Map<String,String> settings(Conv2dConfig c) {
        return Map.of("kernelH",Integer.toString(c.kernelHeight()),"kernelW",Integer.toString(c.kernelWidth()),
                "strideH",Integer.toString(c.strideHeight()),"strideW",Integer.toString(c.strideWidth()),
                "padH",Integer.toString(c.paddingHeight()),"padW",Integer.toString(c.paddingWidth()),
                "dilationH",Integer.toString(c.dilationHeight()),"dilationW",Integer.toString(c.dilationWidth()),"groups",Integer.toString(c.groups()));
    }

    static Conv2dConfig settings(ComponentSpec s) {
        return new Conv2dConfig(s.intValue("kernelH"),s.intValue("kernelW"),s.intValue("strideH"),s.intValue("strideW"),
                s.intValue("padH"),s.intValue("padW"),s.intValue("dilationH"),s.intValue("dilationW"),s.intValue("groups"));
    }

    static void require(ComponentSpec s, String type, String... additional) {
        var spatial=Stream.of("kernelH","kernelW","strideH","strideW","padH","padW","dilationH","dilationW","groups");
        s.require(type,1,Stream.concat(spatial,Stream.of(additional)).toArray(String[]::new));
        if (!s.children().isEmpty()) { throw new IllegalArgumentException("Convolution definition cannot have children"); }
    }

    public static final class Builder {
        private String name="conv2d";
        private long inputs,outputs,seed;
        private boolean bias=true;
        private Conv2dConfig config=new Conv2dConfig(3,3);
        private Builder() { }
        public Builder name(String value) { ModelBuild.name(value); name=value; return this; }
        public Builder inputChannels(long value) { inputs=value; return this; }
        public Builder outputChannels(long value) { outputs=value; return this; }
        public Builder kernelSize(int value) { return kernelSize(value,value); }
        public Builder kernelSize(int height,int width) { config=new Conv2dConfig(height,width,config.strideHeight(),config.strideWidth(),config.paddingHeight(),config.paddingWidth(),config.dilationHeight(),config.dilationWidth(),config.groups()); return this; }
        public Builder stride(int value) { return stride(value,value); }
        public Builder stride(int height,int width) { config=config.stride(height,width); return this; }
        public Builder padding(int value) { return padding(value,value); }
        public Builder padding(int height,int width) { config=config.padding(height,width); return this; }
        public Builder dilation(int value) { return dilation(value,value); }
        public Builder dilation(int height,int width) { config=config.dilation(height,width); return this; }
        public Builder groups(int value) { config=config.groups(value); return this; }
        public Builder bias(boolean value) { bias=value; return this; }
        public Builder seed(long value) { seed=value; return this; }
        public Definition define() { return new Definition(name,inputs,outputs,config,bias,seed); }
        public Conv2d build(ExecutionContext context) { return define().build(context); }
    }
}

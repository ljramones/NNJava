package org.dynamisengine.nn.nn;

import java.util.Map;
import java.util.Objects;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.ops.Pool2dConfig;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Tensor;

/** Channel-first max or average pooling. Padding is excluded from averages. */
public final class Pool2d extends AbstractModule implements UnaryModule {
    public enum Kind { MAX, AVERAGE }
    private final Kind kind;
    private final Pool2dConfig config;

    public Pool2d(ExecutionContext context, Kind kind, Pool2dConfig config) {
        super(context);
        this.kind=Objects.requireNonNull(kind); this.config=Objects.requireNonNull(config);
        ModelBuild.context(context);
        ModelBuild.require(context,"pool2d",DType.FP32,kind==Kind.MAX ? Operation.MAX_POOL2D : Operation.AVG_POOL2D,
                kind==Kind.MAX ? Operation.MAX_POOL2D_BACKWARD : Operation.AVG_POOL2D_BACKWARD);
    }

    @Override public Tensor forward(Tensor input) {
        ensureOpen(); context.requireContext(input);
        return kind==Kind.MAX ? Ops.maxPool2d(input,config) : Ops.avgPool2d(input,config);
    }

    @Override protected Map<String,String> localConfiguration() {
        return Map.of("kind",kind.name(),"window",Integer.toString(config.window()),"stride",Integer.toString(config.stride()),"padding",Integer.toString(config.padding()));
    }
}

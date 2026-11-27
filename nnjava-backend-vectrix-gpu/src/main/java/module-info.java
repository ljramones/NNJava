/**
 * Optional GPU adapter; applications supply their selected native runtime.
 */
module org.dynamisengine.nn.backend.gpu {
    requires transitive org.dynamisengine.nn.core;
    requires transitive org.dynamisengine.gpu.api;
    requires org.dynamisengine.gpu.primitives;
    requires org.dynamisengine.vectrix.compute;
    exports org.dynamisengine.nn.backend.gpu;
    uses org.dynamisengine.gpu.api.compute.GpuProvider;
    provides org.dynamisengine.nn.backend.BackendProvider
            with org.dynamisengine.nn.backend.gpu.ExactGpuProvider,
                    org.dynamisengine.nn.backend.gpu.ThroughputGpuProvider;
}

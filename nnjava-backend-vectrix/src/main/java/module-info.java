/**
 * nnjava-backend-vectrix module.
 */
module org.dynamisengine.nn.backend.vectrix {
    requires transitive org.dynamisengine.nn.core;
    requires org.dynamisengine.vectrix;
    requires org.dynamisengine.vectrix.ml;
    exports org.dynamisengine.nn.backend.vectrix;
    provides org.dynamisengine.nn.backend.BackendProvider
            with org.dynamisengine.nn.backend.vectrix.VectrixBackendProvider;
}

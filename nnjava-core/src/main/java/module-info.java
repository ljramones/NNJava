/**
 * nnjava-core module.
 */
module org.dynamisengine.nn.core {
    exports org.dynamisengine.nn.tensor;
    exports org.dynamisengine.nn.ops;
    exports org.dynamisengine.nn.autograd;
    exports org.dynamisengine.nn.runtime;
    exports org.dynamisengine.nn.backend;
    exports org.dynamisengine.nn.state;
    exports org.dynamisengine.nn.model;
    uses org.dynamisengine.nn.backend.BackendProvider;
}

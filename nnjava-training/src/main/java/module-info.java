/**
 * nnjava-training module.
 */
module org.dynamisengine.nn.training {
    requires transitive org.dynamisengine.nn.core;
    requires transitive org.dynamisengine.nn.layers;
    requires transitive org.dynamisengine.nn.data;
    requires transitive org.dynamisengine.nn.io;
    exports org.dynamisengine.nn.training;
}

/**
 * nnjava module.
 */
module org.dynamisengine.nn.starter {
    requires transitive org.dynamisengine.nn.core;
    requires transitive org.dynamisengine.nn.backend.vectrix;
    requires transitive org.dynamisengine.nn.layers;
    requires transitive org.dynamisengine.nn.transformer;
    requires transitive org.dynamisengine.nn.generation;
    requires transitive org.dynamisengine.nn.data;
    requires transitive org.dynamisengine.nn.io;
    requires transitive org.dynamisengine.nn.training;
}

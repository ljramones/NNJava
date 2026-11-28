/** Explicit, optional pretrained architecture adapters. No numerical, tokenizer or training provider. */
module org.dynamisengine.nn.pretrained {
    requires transitive org.dynamisengine.nn.io;
    requires transitive org.dynamisengine.nn.transformer;
    exports org.dynamisengine.nn.pretrained;
}

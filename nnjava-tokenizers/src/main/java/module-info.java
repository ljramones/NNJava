module org.dynamisengine.nn.tokenizers {
    exports org.dynamisengine.nn.tokenizers;
    uses org.dynamisengine.nn.tokenizers.TokenizerProvider;
    provides org.dynamisengine.nn.tokenizers.TokenizerProvider with org.dynamisengine.nn.tokenizers.ByteTokenizerProvider;
}

module org.dynamisengine.nn.tokenizers {
    requires org.dynamisengine.nn.formats;
    exports org.dynamisengine.nn.tokenizers;
    uses org.dynamisengine.nn.tokenizers.TokenizerProvider;
    provides org.dynamisengine.nn.tokenizers.TokenizerProvider with org.dynamisengine.nn.tokenizers.ByteTokenizerProvider,
            org.dynamisengine.nn.tokenizers.Gpt2TokenizerProvider;
}

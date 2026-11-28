package org.dynamisengine.nn.tokenizers;

import java.util.Map;

/** Service provider for optional tokenizer formats; assets and compatibility policy belong to the provider. */
public interface TokenizerProvider {
    String id();
    Tokenizer create(Map<String,String> configuration);
}

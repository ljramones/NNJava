package org.dynamisengine.nn.pretrained;

import org.dynamisengine.nn.formats.Safetensors;
import org.dynamisengine.nn.formats.TensorFile;
import org.dynamisengine.nn.io.WeightLoader;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.transformer.CausalLanguageModel;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Local standard Llama safetensors checkpoint; no downloads, remote code or tokenizer assumptions.
 */
public final class LlamaCheckpoint {
    private final Path directory;
    private final LlamaConfig configuration;

    private LlamaCheckpoint(Path directory, LlamaConfig configuration) {
        this.directory = directory;
        this.configuration = configuration;
    }

    public static LlamaCheckpoint open(Path directory) throws IOException {
        return new LlamaCheckpoint(directory.toAbsolutePath().normalize(), LlamaConfig.read(directory.resolve("config.json")));
    }

    /**
     * Explicit HF names, with no transpose: both layouts are [out,in]. Extra source tensors are rejected.
     */
    public static WeightLoader weightLoader(LlamaConfig config, TensorFile file) {
        var builder = WeightLoader.builder().expandFloatingPointToFp32(true)
                .map("model.embed_tokens.weight", "embedding.weight").map("model.norm.weight", "finalNorm.weight");
        for (int i = 0; i < config.blockCount(); i++) {
            String source = "model.layers." + i + ".", target = "block" + i + ".";
            builder.map(source + "input_layernorm.weight", target + "attentionNorm.weight")
                    .map(source + "post_attention_layernorm.weight", target + "feedForwardNorm.weight");
            for (String[] pair : new String[][]{{"q_proj", "query"}, {"k_proj", "key"}, {"v_proj", "value"}, {"o_proj", "output"}}) {
                builder.map(source + "self_attn." + pair[0] + ".weight", target + "attention." + pair[1] + ".weight");
            }
            for (String projection : new String[]{"gate", "up", "down"}) {
                builder.map(source + "mlp." + projection + "_proj.weight", target + "feedForward." + projection + ".weight");
            }
        }
        if (!config.tiedEmbeddings()) {
            builder.map("lm_head.weight", "output.weight");
        } else if (file.tensors().containsKey("lm_head.weight")) {
            var embedding = file.tensors().get("model.embed_tokens.weight");
            var head = file.tensors().get("lm_head.weight");
            if (embedding == null || head.type() != embedding.type() || !head.shape().equals(embedding.shape())
                    || file.data("lm_head.weight").mismatch(file.data("model.embed_tokens.weight")) != -1) {
                throw new IllegalArgumentException("Tied checkpoint head differs from embedding storage");
            }
            builder.ignoreSource("lm_head.weight");
        }
        return builder.build();
    }

    public LlamaConfig configuration() {
        return configuration;
    }

    public Path directory() {
        return directory;
    }

    /**
     * Returns an owned model in its default training mode. The caller explicitly selects evaluation mode.
     */
    public CausalLanguageModel load(ExecutionContext context, int contextCapacity) throws IOException {
        var definition = configuration.definition(contextCapacity);
        try (var file = openWeights()) {
            var loader = weightLoader(configuration, file);
            return loader.build(file, definition, context);
        }
    }

    public TensorFile openWeights() throws IOException {
        Path single = directory.resolve("model.safetensors"), index = directory.resolve("model.safetensors.index.json");
        if (Files.exists(single) == Files.exists(index)) {
            throw new IOException("Expected exactly one model.safetensors or model.safetensors.index.json");
        }
        return Files.exists(single) ? Safetensors.open(single) : Safetensors.openIndex(index);
    }
}

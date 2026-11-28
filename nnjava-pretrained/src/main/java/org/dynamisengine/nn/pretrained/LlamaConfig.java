package org.dynamisengine.nn.pretrained;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import org.dynamisengine.nn.formats.JsonDocuments;
import org.dynamisengine.nn.nn.NormalizationKind;
import org.dynamisengine.nn.transformer.CausalLanguageModel;

/** Standard Hugging Face Llama architecture: full split-half RoPE, RMSNorm, bias-free SwiGLU. */
public record LlamaConfig(int vocabularySize, int modelWidth, int blockCount, int contextLength,
                          int queryHeads, int keyValueHeads, int hiddenWidth, float epsilon,
                          double rotaryBase, boolean tiedEmbeddings, Integer bosToken,
                          List<Integer> eosTokenIds) {
    public LlamaConfig {
        eosTokenIds = List.copyOf(eosTokenIds);
        if (bosToken != null && (bosToken < 0 || bosToken >= vocabularySize)
                || eosTokenIds.stream().anyMatch(id -> id < 0 || id >= vocabularySize)) {
            throw new IllegalArgumentException("Control token outside vocabulary");
        }
        // Keep constructor validation identical to the model definition, before allocating any tensors.
        definition(vocabularySize, modelWidth, blockCount, contextLength, queryHeads, keyValueHeads,
                hiddenWidth, epsilon, rotaryBase, tiedEmbeddings);
    }

    public OptionalInt bosTokenId() { return bosToken == null ? OptionalInt.empty() : OptionalInt.of(bosToken); }
    public static LlamaConfig read(Path path) throws IOException { return fromJson(JsonDocuments.read(path)); }

    /** Rejects unsupported architecture switches instead of silently applying approximate defaults. */
    public static LlamaConfig fromJson(Map<String, Object> json) throws IOException {
        try {
            require(json, "model_type", "llama");
            if (json.containsKey("architectures") && !json.get("architectures").equals(List.of("LlamaForCausalLM"))) {
                throw new IOException("Expected LlamaForCausalLM architecture");
            }
            requireDefault(json, "hidden_act", "silu");
            for (String key : List.of("attention_bias", "mlp_bias", "rope_interleaved", "use_sliding_window", "alibi")) {
                requireDefault(json, key, false);
            }
            for (String key : List.of("rope_scaling", "rope_parameters", "sliding_window", "auto_map")) {
                if (json.get(key) != null) { throw new IOException("Unsupported Llama option: " + key); }
            }
            if (integer(json, "pretraining_tp", 1) != 1 || number(json, "partial_rotary_factor", 1) != 1
                    || number(json, "attention_dropout", 0) != 0 || number(json, "hidden_dropout", 0) != 0) {
                throw new IOException("Unsupported Llama partition, partial RoPE or dropout configuration");
            }
            int width = integer(json, "hidden_size"), heads = integer(json, "num_attention_heads");
            if (heads <= 0 || width <= 0 || width % heads != 0 || integer(json, "head_dim", width / heads) != width / heads) {
                throw new IOException("Unsupported Llama head dimensions");
            }
            var eos = new ArrayList<Integer>(); Object eosValue = json.get("eos_token_id");
            if (eosValue instanceof List<?> values) {
                for (var value : values) { eos.add(Math.toIntExact(JsonDocuments.integer(value, "eos_token_id"))); }
            } else if (eosValue != null) { eos.add(Math.toIntExact(JsonDocuments.integer(eosValue, "eos_token_id"))); }
            return new LlamaConfig(integer(json, "vocab_size"), width, integer(json, "num_hidden_layers"),
                    integer(json, "max_position_embeddings"), heads, integer(json, "num_key_value_heads", heads),
                    integer(json, "intermediate_size"), (float) number(json, "rms_norm_eps", 1e-6),
                    number(json, "rope_theta", 10_000), bool(json, "tie_word_embeddings", false),
                    json.get("bos_token_id") == null ? null : integer(json, "bos_token_id"), eos);
        } catch (IllegalArgumentException | ArithmeticException failure) { throw new IOException("Invalid Llama configuration", failure); }
    }

    public CausalLanguageModel.Definition definition() { return definition(contextLength); }

    /** Caps allocated cache capacity without changing position frequencies or weight geometry. */
    public CausalLanguageModel.Definition definition(int capacity) {
        if (capacity < 1 || capacity > contextLength) { throw new IllegalArgumentException("Context capacity exceeds checkpoint configuration"); }
        return definition(vocabularySize, modelWidth, blockCount, capacity, queryHeads, keyValueHeads,
                hiddenWidth, epsilon, rotaryBase, tiedEmbeddings);
    }

    private static CausalLanguageModel.Definition definition(int vocabulary, int width, int blocks, int context,
                                                             int heads, int kvHeads, int hidden, float epsilon,
                                                             double base, boolean tied) {
        return CausalLanguageModel.builder().name("llama").vocabularySize(vocabulary).modelWidth(width)
                .blockCount(blocks).contextLength(context).normalization(NormalizationKind.RMS)
                .normalizationEpsilon(epsilon).tieInputOutputEmbeddings(tied)
                .attention(a -> a.queryHeads(heads).keyValueHeads(kvHeads).bias(false).rotaryBase(base))
                .feedForward(f -> f.hiddenWidth(hidden).swiGlu()).define();
    }
    private static void require(Map<String, Object> json, String key, Object expected) throws IOException {
        if (!expected.equals(json.get(key))) { throw new IOException("Unsupported " + key + ": " + json.get(key)); }
    }
    private static void requireDefault(Map<String, Object> json, String key, Object expected) throws IOException {
        if (json.containsKey(key)) { require(json, key, expected); }
    }
    private static int integer(Map<String, Object> json, String key) throws IOException { return Math.toIntExact(JsonDocuments.integer(json.get(key), key)); }
    private static int integer(Map<String, Object> json, String key, int fallback) throws IOException { return json.containsKey(key) ? integer(json, key) : fallback; }
    private static double number(Map<String, Object> json, String key, double fallback) throws IOException {
        if (!json.containsKey(key)) { return fallback; }
        if (!(json.get(key) instanceof Number value) || !Double.isFinite(value.doubleValue())) { throw new IOException("Expected finite number: " + key); }
        return value.doubleValue();
    }
    private static boolean bool(Map<String, Object> json, String key, boolean fallback) throws IOException {
        if (!json.containsKey(key)) { return fallback; }
        if (!(json.get(key) instanceof Boolean value)) { throw new IOException("Expected boolean: " + key); } return value;
    }
}

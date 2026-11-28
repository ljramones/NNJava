package org.dynamisengine.nn.pretrained;

import org.dynamisengine.nn.formats.JsonDocuments;
import org.dynamisengine.nn.nn.NormalizationKind;
import org.dynamisengine.nn.transformer.CausalLanguageModel;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class LlamaConfigTest {
    private static Map<String, Object> json() throws IOException {
        return new LinkedHashMap<>(JsonDocuments.parse(("{\"model_type\":\"llama\",\"architectures\":[\"LlamaForCausalLM\"],"
                + "\"vocab_size\":19,\"hidden_size\":8,\"num_hidden_layers\":2,\"intermediate_size\":12,"
                + "\"max_position_embeddings\":32,\"num_attention_heads\":2,\"num_key_value_heads\":1,"
                + "\"bos_token_id\":0,\"eos_token_id\":[1,2],\"tie_word_embeddings\":true}").getBytes(StandardCharsets.UTF_8)));
    }

    @Test
    void mapsStandardLlamaGeometryAndControlsToSerializableDefinitions() throws Exception {
        var config = LlamaConfig.fromJson(json());
        var definition = config.definition(16);
        assertEquals(2, definition.blockCount());
        assertEquals(16, definition.contextLength());
        assertEquals(1, definition.block().attention().kvHeads());
        assertEquals(NormalizationKind.RMS, definition.block().normalization());
        assertTrue(definition.tieInputOutputEmbeddings());
        assertTrue(definition.block().attention().rotary());
        assertEquals(0, config.bosTokenId().orElseThrow());
        assertEquals(List.of(1, 2), config.eosTokenIds());
        assertEquals(definition, CausalLanguageModel.Definition.fromSpec(definition.specification()));
        assertThrows(IllegalArgumentException.class, () -> config.definition(33));
    }

    @Test
    void omittedOptionalFieldsUseStandardLlamaDefaults() throws Exception {
        var json = json();
        json.remove("num_key_value_heads");
        json.remove("bos_token_id");
        json.remove("eos_token_id");
        json.remove("tie_word_embeddings");
        var config = LlamaConfig.fromJson(json);
        assertEquals(2, config.keyValueHeads());
        assertFalse(config.tiedEmbeddings());
        assertTrue(config.bosTokenId().isEmpty());
        assertTrue(config.eosTokenIds().isEmpty());
        assertEquals(1e-6f, config.epsilon());
        assertEquals(10000, config.rotaryBase());
    }

    @Test
    void unsupportedArchitecturalSwitchesFailInsteadOfChangingTheModelSilently() throws Exception {
        for (var entry : Map.<String, Object>ofEntries(Map.entry("model_type", "mistral"), Map.entry("hidden_act", "gelu"),
                Map.entry("attention_bias", true), Map.entry("mlp_bias", true), Map.entry("rope_interleaved", true),
                Map.entry("rope_scaling", Map.of("factor", 2)), Map.entry("sliding_window", 4096L),
                Map.entry("head_dim", 2L), Map.entry("pretraining_tp", 2L), Map.entry("partial_rotary_factor", .5),
                Map.entry("attention_dropout", .1), Map.entry("auto_map", Map.of("AutoModel", "remote.py")),
                Map.entry("architectures", List.of("OtherModel"))).entrySet()) {
            var json = json();
            json.put(entry.getKey(), entry.getValue());
            assertThrows(IOException.class, () -> LlamaConfig.fromJson(json), entry.getKey());
        }
    }

    @Test
    void invalidTypesDimensionsAndTokenIdsFailDuringConfigurationParsing() throws Exception {
        for (var entry : Map.<String, Object>of("hidden_size", 7L, "num_attention_heads", 0L, "vocab_size", -1L,
                "num_hidden_layers", 0L, "intermediate_size", 0L, "rms_norm_eps", Double.NaN,
                "eos_token_id", List.of(19L), "bos_token_id", -1L, "tie_word_embeddings", "true").entrySet()) {
            var json = json();
            json.put(entry.getKey(), entry.getValue());
            assertThrows(IOException.class, () -> LlamaConfig.fromJson(json), entry.getKey());
        }
        var json = json();
        json.remove("hidden_size");
        assertThrows(IOException.class, () -> LlamaConfig.fromJson(json));
    }
}

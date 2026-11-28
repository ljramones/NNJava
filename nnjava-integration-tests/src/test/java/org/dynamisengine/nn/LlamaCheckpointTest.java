package org.dynamisengine.nn;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import org.dynamisengine.nn.backend.Backend;
import org.dynamisengine.nn.backend.KernelCall;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.backend.vectrix.VectrixBackend;
import org.dynamisengine.nn.formats.JsonDocuments;
import org.dynamisengine.nn.formats.Safetensors;
import org.dynamisengine.nn.formats.TensorDescriptor;
import org.dynamisengine.nn.io.StateDict;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.pretrained.LlamaCheckpoint;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class LlamaCheckpointTest {
    @TempDir Path directory;

    @Test void tiedAndUntiedBf16CheckpointsMatchIndependentTransformersLogits() throws Exception {
        for (boolean tied : new boolean[]{true, false}) {
            var checkpoint = LlamaCheckpoint.open(fixture(tied)); var reference = JsonDocuments.read(fixture(tied).resolve("reference.json"));
            int[] ids = JsonDocuments.array(reference.get("input_ids"), "ids").stream().mapToInt(v -> ((Number) v).intValue()).toArray();
            float[] expected = flatten(reference.get("logits"));
            try (var context = ExecutionContext.cpu()) {
                try (var model = checkpoint.load(context, 16); var input = context.tensor(ids, Shape.of(1, ids.length))) {
                    assertTrue(model.isTraining()); model.training(false);
                    assertEquals(tied, StateDict.capture(model).aliases().containsKey("output.weight"));
                    try (var actual = model.forward(input)) { assertArrayEquals(expected, actual.toFloatArray(), 3e-6f); }
                    int baseline = context.liveAllocations();
                    try (var session = model.newSession(1); var prefix = Ops.slice(input, 1, 0, 2); var suffix = Ops.slice(input, 1, 2, 4);
                         var first = session.prefill(prefix); var second = session.decode(suffix)) {
                        assertArrayEquals(java.util.Arrays.copyOfRange(expected, 0, 38), first.toFloatArray(), 3e-6f);
                        assertArrayEquals(java.util.Arrays.copyOfRange(expected, 38, 76), second.toFloatArray(), 3e-6f);
                        assertEquals(4, session.length());
                    }
                    assertEquals(baseline, context.liveAllocations());
                }
                assertEquals(0, context.liveAllocations());
            }
        }
    }

    @Test void missingEarlyAndLateTensorsCloseTheEntireOwnedModel() throws Exception {
        for (String omitted : List.of("model.embed_tokens.weight", "model.layers.1.mlp.down_proj.weight")) {
            prepare(true); rewriteWeights(true, omitted, 0);
            try (var context = ExecutionContext.cpu()) {
                assertThrows(IllegalArgumentException.class, () -> LlamaCheckpoint.open(directory).load(context, 16));
                assertEquals(0, context.liveAllocations());
            }
        }
    }

    @Test void tiedHeadDuplicatesAreVerifiedBeforeTheyAreIgnored() throws Exception {
        prepare(true); rewriteWeights(true, null, 1);
        try (var context = ExecutionContext.cpu(); var model = LlamaCheckpoint.open(directory).load(context, 16)) {
            assertEquals("embedding.weight", StateDict.capture(model).aliases().get("output.weight"));
        }
        rewriteWeights(true, null, 2);
        try (var context = ExecutionContext.cpu()) {
            assertThrows(IllegalArgumentException.class, () -> LlamaCheckpoint.open(directory).load(context, 16));
            assertEquals(0, context.liveAllocations());
        }
    }

    @Test void untiedHeadIsRequiredAndAmbiguousOrMissingArchivesFailWithoutAllocations() throws Exception {
        prepare(false); rewriteWeights(false, "lm_head.weight", 0);
        try (var context = ExecutionContext.cpu()) {
            assertThrows(IllegalArgumentException.class, () -> LlamaCheckpoint.open(directory).load(context, 16)); assertEquals(0, context.liveAllocations());
            Files.delete(directory.resolve("model.safetensors"));
            assertThrows(IOException.class, () -> LlamaCheckpoint.open(directory).load(context, 16)); assertEquals(0, context.liveAllocations());
            Files.copy(fixture(false).resolve("model.safetensors"), directory.resolve("model.safetensors"));
            Files.writeString(directory.resolve("model.safetensors.index.json"), "{}");
            assertThrows(IOException.class, () -> LlamaCheckpoint.open(directory).load(context, 16)); assertEquals(0, context.liveAllocations());
        }
    }

    @Test void assignmentFailureReleasesEveryParameterOfTheUnreturnedModel() throws Exception {
        var underlying = new VectrixBackend(); var copies = new AtomicInteger();
        var backend = (Backend) Proxy.newProxyInstance(getClass().getClassLoader(), new Class<?>[]{Backend.class}, (proxy, method, args) -> {
            if (method.getName().equals("execute") && args[0] instanceof KernelCall call && call.operation() == Operation.COPY && copies.incrementAndGet() == 2) {
                throw new IllegalStateException("Injected second weight assignment failure");
            }
            try { return method.invoke(underlying, args); } catch (InvocationTargetException failure) { throw failure.getCause(); }
        });
        var context = ExecutionContext.create(backend);
        try {
            assertThrows(IllegalStateException.class, () -> LlamaCheckpoint.open(fixture(true)).load(context, 16));
            assertEquals(0, context.liveAllocations());
        } finally { assertThrows(IllegalStateException.class, context::close); } // The context retains backend execution failure.
    }

    private static Path fixture(boolean tied) throws Exception {
        return Path.of(LlamaCheckpointTest.class.getResource("/pretrained/" + (tied ? "tied-llama" : "untied-llama")).toURI());
    }
    private void prepare(boolean tied) throws Exception {
        Files.copy(fixture(tied).resolve("config.json"), directory.resolve("config.json"), java.nio.file.StandardCopyOption.REPLACE_EXISTING);
    }
    private void rewriteWeights(boolean tied, String omitted, int duplicateHead) throws Exception {
        var descriptors = new LinkedHashMap<String, TensorDescriptor>(); var payloads = new LinkedHashMap<String, byte[]>();
        try (var source = Safetensors.open(fixture(tied).resolve("model.safetensors"))) {
            for (var entry : source.tensors().entrySet()) {
                if (!entry.getKey().equals(omitted)) { descriptors.put(entry.getKey(), entry.getValue()); payloads.put(entry.getKey(), source.data(entry.getKey()).toArray(ValueLayout.JAVA_BYTE)); }
            }
            if (duplicateHead > 0) {
                descriptors.put("lm_head.weight", descriptors.get("model.embed_tokens.weight"));
                byte[] bytes = payloads.get("model.embed_tokens.weight").clone(); if (duplicateHead == 2) { bytes[0] ^= 1; } payloads.put("lm_head.weight", bytes);
            }
        }
        var header = new StringBuilder("{"); var bytes = new ByteArrayOutputStream();
        for (var entry : descriptors.entrySet()) {
            if (header.length() > 1) { header.append(','); } int start = bytes.size(); bytes.writeBytes(payloads.get(entry.getKey()));
            header.append('"').append(entry.getKey()).append("\":{\"dtype\":\"").append(entry.getValue().type()).append("\",\"shape\":")
                    .append(entry.getValue().shape()).append(",\"data_offsets\":[").append(start).append(',').append(bytes.size()).append("]}");
        }
        byte[] json = header.append('}').toString().getBytes(StandardCharsets.UTF_8);
        Files.write(directory.resolve("model.safetensors"), ByteBuffer.allocate(8 + json.length + bytes.size()).order(ByteOrder.LITTLE_ENDIAN)
                .putLong(json.length).put(json).put(bytes.toByteArray()).array());
    }
    private static float[] flatten(Object value) throws IOException {
        var rows = JsonDocuments.array(value, "logits"); var result = new java.util.ArrayList<Float>();
        for (var row : rows) { for (var number : JsonDocuments.array(row, "row")) { result.add(((Number) number).floatValue()); } }
        float[] values = new float[result.size()]; for (int i = 0; i < values.length; i++) { values[i] = result.get(i); } return values;
    }
}

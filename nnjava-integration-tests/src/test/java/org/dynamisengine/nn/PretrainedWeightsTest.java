package org.dynamisengine.nn;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.dynamisengine.nn.formats.*;
import org.dynamisengine.nn.generation.GenerationSession;
import org.dynamisengine.nn.io.StateDict;
import org.dynamisengine.nn.io.WeightLoader;
import org.dynamisengine.nn.nn.Linear;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.state.TensorSnapshot;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.transformer.CausalLanguageModel;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class PretrainedWeightsTest {
    @TempDir Path directory;

    @Test void explicitTransposedMappingMatchesIndependentDenseProjection() throws Exception {
        Path path = safe(Map.of("external.kernel", new TensorSnapshot(Shape.of(3, 2), new float[]{1, 4, 2, 5, 3, 6}),
                "external.bias", new TensorSnapshot(Shape.of(2), new float[]{.5f, -1})));
        var loader = WeightLoader.builder().transpose("external.kernel", "weight").map("external.bias", "bias").build();
        try (var context = ExecutionContext.cpu(); var model = new Linear(context, 3, 2, 42); var source = Safetensors.open(path)) {
            int before = context.liveAllocations(); long version = model.namedParameters().get("weight").mutationVersion();
            var report = loader.validate(source, model); assertEquals(version, model.namedParameters().get("weight").mutationVersion());
            assertEquals(2, report.uniqueParameters()); assertEquals(32, report.sourceBytes());
            loader.load(source, model); assertTrue(model.namedParameters().get("weight").mutationVersion() > version);
            try (var input = context.tensor(new float[]{1, 2, 3}, Shape.of(1, 3)); var output = model.forward(input)) {
                assertArrayEquals(new float[]{14.5f, 31}, output.toFloatArray(), 1e-6f);
            }
            assertEquals(before, context.liveAllocations()); assertTrue(model.isTraining());
        }
    }

    @Test void allMappingFailuresPrecedeTheFirstParameterMutation() throws Exception {
        Path path = safe(Map.of("kernel", new TensorSnapshot(Shape.of(2, 3), new float[]{1, 2, 3, 4, 5, 6}),
                "bias", new TensorSnapshot(Shape.of(2), new float[]{7, 8}), "unused", new TensorSnapshot(Shape.of(1), new float[]{9})));
        try (var context = ExecutionContext.cpu(); var model = new Linear(context, 3, 2, 7); var source = Safetensors.open(path)) {
            var before = StateDict.capture(model); long version = model.namedParameters().get("weight").mutationVersion();
            var bad = List.of(
                    WeightLoader.builder().map("kernel", "weight").build(),
                    WeightLoader.builder().map("kernel", "weight").map("bias", "bias").build(),
                    WeightLoader.builder().map("kernel", "weight").map("bias", "absent").requireAllSources(false).build(),
                    WeightLoader.builder().map("kernel", "weight").map("unused", "bias").ignoreSource("bias").build(),
                    WeightLoader.builder().map("kernel", "weight").map("bias", "bias").ignoreSource("unused").maxTensorBytes(8).build(),
                    WeightLoader.builder().map("kernel", "weight").map("bias", "bias").ignoreSource("unknown").build());
            for (var loader : bad) {
                assertThrows(IllegalArgumentException.class, () -> loader.load(source, model));
                assertEquals(version, model.namedParameters().get("weight").mutationVersion());
                before.tensors().forEach((name, value) -> assertArrayEquals(value.floats(), StateDict.capture(model).tensors().get(name).floats()));
            }
            var report = WeightLoader.builder().map("kernel", "weight").map("bias", "bias").requireAllSources(false).build().load(source, model);
            assertEquals(java.util.Set.of("unused"), report.ignoredSources());
        }
    }

    @Test void fp16AndBf16ExpansionIsExplicitAndPreservesIeeeValues() throws Exception {
        for (DType type : new DType[]{DType.FP16, DType.BF16}) {
            short[] values = type == DType.FP16 ? new short[]{0x3c00, (short) 0xc000, (short) 0x8000, 0x0001}
                    : new short[]{0x3f80, (short) 0xc000, (short) 0x8000, 0x0001};
            Path path = safe(Map.of("w", new TensorSnapshot(Shape.of(1, 4), type, values)));
            try (var context = ExecutionContext.cpu(); var model = new Linear(context, 4, 1, 0, false); var source = Safetensors.open(path)) {
                assertThrows(IllegalArgumentException.class, () -> WeightLoader.builder().map("w", "weight").build().load(source, model));
                WeightLoader.builder().map("w", "weight").expandFloatingPointToFp32(true).build().load(source, model);
                try (var tensor = model.namedParameters().get("weight").value()) {
                    float[] actual = tensor.toFloatArray(); assertEquals(1, actual[0]); assertEquals(-2, actual[1]);
                    assertEquals(0x80000000, Float.floatToRawIntBits(actual[2]));
                    assertEquals(type == DType.FP16 ? 0x33800000 : 0x00010000, Float.floatToRawIntBits(actual[3]));
                }
            }
        }
    }

    @Test void lowPrecisionIntegerAndBooleanWeightsRetainTheirStorageBits() throws Exception {
        for (var snapshot : List.of(new TensorSnapshot(Shape.of(2), DType.FP16, new short[]{(short) 0x7e31, (short) 0x8000}),
                new TensorSnapshot(Shape.of(2), DType.BF16, new short[]{(short) 0x7fc1, (short) 0xffff}),
                new TensorSnapshot(Shape.of(2), new int[]{Integer.MIN_VALUE, Integer.MAX_VALUE}),
                new TensorSnapshot(Shape.of(2), new boolean[]{true, false}))) {
            try (var context = ExecutionContext.cpu(); var source = Safetensors.open(safe(Map.of("p", snapshot)))) {
                var imported = WeightLoader.readTensor(source, "p");
                assertEquals(snapshot.dtype(), imported.dtype());
                try (var tensor = imported.materialize(context)) { assertTrue(java.util.Objects.deepEquals(snapshot.valuesCopy(), TensorSnapshot.capture(tensor).valuesCopy())); }
            }
        }
    }

    @Test void tiedDecoderWeightsLoadOnceAndReproduceCachedGeneration() throws Exception {
        var definition = CausalLanguageModel.builder().vocabularySize(7).modelWidth(8).blockCount(2).contextLength(16)
                .attention(a -> a.queryHeads(2).keyValueHeads(1)).feedForward(f -> f.hiddenWidth(12)).seed(42).define();
        try (var context = ExecutionContext.cpu(); var original = definition.build(context); var restored = definition.build(context)) {
            var snapshots = StateDict.capture(original); var external = new LinkedHashMap<String, TensorSnapshot>();
            var builder = WeightLoader.builder(); snapshots.tensors().forEach((name, value) -> { external.put("checkpoint." + name, value); builder.map("checkpoint." + name, name); });
            try (var source = Safetensors.open(safe(external))) {
                var report = builder.build().load(source, restored);
                assertEquals(snapshots.aliases(), report.aliases()); assertFalse(report.aliases().isEmpty());
                assertEquals(snapshots.tensors().size(), report.uniqueParameters());
            }
            original.training(false); restored.training(false);
            try (var input = context.tensor(new int[]{1, 2, 3}, Shape.of(1, 3)); var expected = original.forward(input); var actual = restored.forward(input)) {
                assertArrayEquals(expected.toFloatArray(), actual.toFloatArray());
            }
            try (var a = GenerationSession.builder().model(original).maxNewTokens(3).sampling(s -> s.temperature(0)).build();
                 var b = GenerationSession.builder().model(restored).maxNewTokens(3).sampling(s -> s.temperature(0)).build()) {
                assertArrayEquals(a.generate(new int[]{1, 2}).tokenIds(), b.generate(new int[]{1, 2}).tokenIds());
            }
        }
    }

    @Test void rejectsConflictingTiedMappingsAndLoadingDuringAnActiveTape() throws Exception {
        var definition = CausalLanguageModel.builder().vocabularySize(3).modelWidth(4).blockCount(1).contextLength(8)
                .attention(a -> a.queryHeads(1)).feedForward(f -> f.hiddenWidth(8)).define();
        try (var context = ExecutionContext.cpu(); var model = definition.build(context)) {
            var state = StateDict.capture(model); var builder = WeightLoader.builder(); state.tensors().keySet().forEach(name -> builder.map(name, name));
            try (var source = Safetensors.open(safe(state.tensors()))) {
                var alias = state.aliases().entrySet().iterator().next();
                var conflicting = WeightLoader.builder(); state.tensors().keySet().forEach(name -> conflicting.map(name, name));
                conflicting.map(alias.getValue(), alias.getKey());
                assertThrows(IllegalArgumentException.class, () -> conflicting.build().load(source, model));
                try (var tape = context.gradientTape()) { assertThrows(IllegalStateException.class, () -> builder.build().load(source, model)); }
            }
        }
    }

    @Test void invalidBooleanPayloadIsRejectedWhenReadingAHostSnapshot() throws Exception {
        var snapshot = new TensorSnapshot(Shape.of(2), new boolean[]{true, false}); Path path = safe(Map.of("p", snapshot));
        byte[] bytes = Files.readAllBytes(path); bytes[bytes.length - 1] = 2; Files.write(path, bytes);
        try (var source = Safetensors.open(path)) {
            assertThrows(IllegalArgumentException.class, () -> WeightLoader.readTensor(source, "p"));
        }
    }

    @Test void backendFailureReleasesStagedTensorAndLeavesModelExplicitlyPartial() throws Exception {
        Path path = safe(Map.of("w", new TensorSnapshot(Shape.of(1, 2), new float[]{1, 2}), "b", new TensorSnapshot(Shape.of(1), new float[]{3})));
        var countdown = new java.util.concurrent.atomic.AtomicInteger(-1); var underlying = new org.dynamisengine.nn.backend.vectrix.VectrixBackend();
        var backend = (org.dynamisengine.nn.backend.Backend) java.lang.reflect.Proxy.newProxyInstance(getClass().getClassLoader(),
                new Class<?>[]{org.dynamisengine.nn.backend.Backend.class}, (proxy, method, args) -> {
                    if (method.getName().equals("allocate") && countdown.get() > 0 && countdown.decrementAndGet() == 0) { throw new IllegalStateException("Injected allocation failure"); }
                    try { return method.invoke(underlying, args); } catch (java.lang.reflect.InvocationTargetException error) { throw error.getCause(); }
                });
        try (var context = ExecutionContext.create(backend); var model = new Linear(context, 2, 1, 7); var source = Safetensors.open(path)) {
            int before = context.liveAllocations(); countdown.set(2); long biasVersion = model.namedParameters().get("bias").mutationVersion();
            assertThrows(IllegalStateException.class, () -> WeightLoader.builder().map("w", "weight").map("b", "bias").build().load(source, model));
            assertEquals(before, context.liveAllocations());
            try (var weight = model.namedParameters().get("weight").value()) { assertArrayEquals(new float[]{1, 2}, weight.toFloatArray()); }
            assertEquals(biasVersion, model.namedParameters().get("bias").mutationVersion());
        }
    }

    @Test void ggufUsesTheSameMappingsAndRejectsQuantizedAssignmentBeforeMutation() throws Exception {
        for (int type : new int[]{0, 12}) {
            int columns = type == 0 ? 2 : 256;
            var bytes = new ByteArrayOutputStream();
            bytes.writeBytes(ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN).putInt(0x46554747).putInt(3).putLong(1).putLong(0).array());
            byte[] name = "external.weight".getBytes(StandardCharsets.UTF_8);
            bytes.writeBytes(ByteBuffer.allocate(8 + name.length + 4 + 16 + 4 + 8).order(ByteOrder.LITTLE_ENDIAN)
                    .putLong(name.length).put(name).putInt(2).putLong(columns).putLong(1).putInt(type).putLong(0).array());
            while (bytes.size() % 32 != 0) { bytes.write(0); }
            bytes.writeBytes(type == 0 ? ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putFloat(2).putFloat(3).array() : new byte[144]);
            Path path = directory.resolve("model.gguf"); Files.write(path, bytes.toByteArray());
            try (var context = ExecutionContext.cpu(); var model = new Linear(context, columns, 1, 7, false); var source = Gguf.open(path)) {
                var loader = WeightLoader.builder().map("external.weight", "weight").build();
                if (type == 0) {
                    loader.load(source, model);
                    try (var input = context.tensor(new float[]{4, 5}, Shape.of(1, 2)); var output = model.forward(input)) { assertArrayEquals(new float[]{23}, output.toFloatArray()); }
                } else {
                    long version = model.namedParameters().get("weight").mutationVersion();
                    assertThrows(UnsupportedOperationException.class, () -> loader.load(source, model));
                    assertThrows(UnsupportedOperationException.class, () -> WeightLoader.readTensor(source, "external.weight"));
                    assertEquals(version, model.namedParameters().get("weight").mutationVersion());
                }
            }
        }
    }

    @Test void hostSnapshotsOutliveMappedFilesAndRespectSelectionBudgets() throws Exception {
        Path path = safe(Map.of("w", new TensorSnapshot(Shape.of(2), new float[]{3, 7}))); TensorSnapshot snapshot;
        try (var source = Safetensors.open(path)) {
            assertThrows(IllegalArgumentException.class, () -> WeightLoader.readTensor(source, "missing"));
            assertThrows(IllegalArgumentException.class, () -> WeightLoader.readTensor(source, "w", 4));
            snapshot = WeightLoader.readTensor(source, "w");
        }
        try (var context = ExecutionContext.cpu(); var tensor = snapshot.materialize(context)) { assertArrayEquals(new float[]{3, 7}, tensor.toFloatArray()); }
    }

    private Path safe(Map<String, TensorSnapshot> weights) throws IOException {
        var header = new StringBuilder("{"); var data = new ByteArrayOutputStream(); boolean first = true;
        for (var entry : weights.entrySet()) {
            var value = entry.getValue(); String type = switch (value.dtype()) { case FP32 -> "F32"; case FP16 -> "F16"; default -> value.dtype().name().replace("INT32", "I32"); };
            int start = data.size(); var buffer = ByteBuffer.allocate(Math.toIntExact(value.shape().elements() * value.dtype().bytes())).order(ByteOrder.LITTLE_ENDIAN);
            switch (value.dtype()) {
                case FP32 -> { for (float v : value.floats()) { buffer.putFloat(v); } }
                case INT32 -> { for (int v : value.integers()) { buffer.putInt(v); } }
                case BOOL -> { for (boolean v : value.booleans()) { buffer.put((byte) (v ? 1 : 0)); } }
                case FP16, BF16 -> { for (short v : value.shorts()) { buffer.putShort(v); } }
            }
            data.writeBytes(buffer.array()); if (!first) { header.append(','); } first = false;
            header.append('"').append(entry.getKey()).append("\":{\"dtype\":\"").append(type).append("\",\"shape\":")
                    .append(java.util.Arrays.toString(value.shape().dimensions())).append(",\"data_offsets\":[").append(start).append(',').append(data.size()).append("]}");
        }
        byte[] json = header.append('}').toString().getBytes(StandardCharsets.UTF_8);
        Path path = directory.resolve("weights.safetensors"); Files.write(path, ByteBuffer.allocate(8 + json.length + data.size()).order(ByteOrder.LITTLE_ENDIAN)
                .putLong(json.length).put(json).put(data.toByteArray()).array()); return path;
    }
}

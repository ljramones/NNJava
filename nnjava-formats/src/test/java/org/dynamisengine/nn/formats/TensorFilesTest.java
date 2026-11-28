package org.dynamisengine.nn.formats;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class TensorFilesTest {
    @TempDir Path directory;
    private static final String TENSOR = "\"weight\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0,24]}";

    @Test void mapsSafetensorsWithMetadataAndDetachedDescriptors() throws Exception {
        byte[] data = ByteBuffer.allocate(24).order(ByteOrder.LITTLE_ENDIAN).putFloat(1).putFloat(2).putFloat(3).putFloat(4).putFloat(5).putFloat(6).array();
        Path path = safe("weights", "{\"__metadata__\":{\"format\":\"pt\"}," + TENSOR + "}  ", data);
        MemorySegment borrowed; TensorDescriptor descriptor;
        var file = Safetensors.open(path);
        try {
            descriptor = file.tensors().get("weight"); assertEquals(List.of(2L, 3L), descriptor.shape());
            assertEquals("pt", file.metadata().get("format")); borrowed = file.data("weight");
            assertArrayEquals(data, borrowed.toArray(ValueLayout.JAVA_BYTE)); assertTrue(borrowed.isReadOnly());
            assertThrows(UnsupportedOperationException.class, () -> file.tensors().clear());
            assertThrows(IllegalArgumentException.class, () -> file.data("missing"));
        } finally { file.close(); }
        file.close(); assertEquals(6, descriptor.elements());
        assertThrows(IllegalStateException.class, file::metadata);
        assertThrows(IllegalStateException.class, () -> borrowed.get(ValueLayout.JAVA_BYTE, 0));
    }

    @Test void supportsScalarsEmptyShapesAndAllUnquantizedStorageTypes() throws Exception {
        String header = "{\"scalar\":{\"dtype\":\"I32\",\"shape\":[],\"data_offsets\":[0,4]},"
                + "\"empty\":{\"dtype\":\"F32\",\"shape\":[0,2],\"data_offsets\":[4,4]},"
                + "\"half\":{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[4,6]},"
                + "\"brain\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[6,8]},"
                + "\"mask\":{\"dtype\":\"BOOL\",\"shape\":[1],\"data_offsets\":[8,9]}}";
        try (var file = Safetensors.open(safe("types", header, new byte[9]))) {
            assertEquals(1, file.tensors().get("scalar").elements()); assertEquals(0, file.tensors().get("empty").elements());
            assertEquals(TensorType.BF16, file.tensors().get("brain").type()); assertEquals(5, file.tensors().size());
        }
    }

    @Test void rejectsMalformedSafetensorsHeadersAndUnsupportedTypes() throws Exception {
        for (String header : List.of("[]", "{" + TENSOR + "," + TENSOR + "}", "{" + TENSOR.replace("F32", "F64") + "}",
                "{" + TENSOR.replace("[2,3]", "[-2,3]") + "}", "{" + TENSOR.replace("[2,3]", "[9223372036854775807,2]") + "}",
                "{" + TENSOR.replace("[2,3]", "[2.0,3]") + "}", "{" + TENSOR.replace("[0,24]", "[24,0]") + "}",
                "{\"__metadata__\":{\"invalid\":3}," + TENSOR + "}", "{" + TENSOR + "}{}")) {
            assertThrows(IOException.class, () -> Safetensors.open(safe("bad", header, new byte[24])), header);
        }
        Path path = directory.resolve("huge");
        Files.write(path, ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(Long.MAX_VALUE).array());
        assertThrows(IOException.class, () -> Safetensors.open(path));
    }

    @Test void rejectsOverlapHolesTruncationAndTrailingPayloadBytes() throws Exception {
        String duplicateRange = TENSOR + ",\"second\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[20,24]}";
        assertThrows(IOException.class, () -> Safetensors.open(safe("overlap", "{" + duplicateRange + "}", new byte[24])));
        assertThrows(IOException.class, () -> Safetensors.open(safe("hole", "{" + TENSOR.replace("[0,24]", "[4,28]") + "}", new byte[28])));
        assertThrows(IOException.class, () -> Safetensors.open(safe("short", "{" + TENSOR + "}", new byte[23])));
        assertThrows(IOException.class, () -> Safetensors.open(safe("trailing", "{" + TENSOR + "}", new byte[25])));
    }

    @Test void shardIndexesValidateOwnershipAndCloseEveryMappedShard() throws Exception {
        safe("a.safetensors", "{" + TENSOR + "}", new byte[24]);
        safe("b.safetensors", "{\"bias\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8]}}", new byte[8]);
        Path index = directory.resolve("model.safetensors.index.json");
        Files.writeString(index, "{\"metadata\":{\"total_size\":32},\"weight_map\":{\"weight\":\"a.safetensors\",\"bias\":\"b.safetensors\"}}");
        MemorySegment first, second;
        try (var file = Safetensors.openIndex(index)) {
            assertEquals(2, file.tensors().size()); first = file.data("weight"); second = file.data("bias");
            assertEquals(32L, file.metadata().get("total_size"));
        }
        assertFalse(first.scope().isAlive()); assertFalse(second.scope().isAlive());
        for (String mapping : List.of("\"weight\":\"../a.safetensors\"", "\"weight\":\"/tmp/outside\"", "\"absent\":\"a.safetensors\"",
                "\"weight\":\"b.safetensors\"", "\"weight\":\"missing.safetensors\"")) {
            Files.writeString(index, "{\"weight_map\":{" + mapping + "}}"); assertThrows(IOException.class, () -> Safetensors.openIndex(index));
        }
    }

    @Test void rejectsShardSymlinksEscapingTheIndexDirectory() throws Exception {
        Path child = Files.createDirectory(directory.resolve("child"));
        safe("outside", "{" + TENSOR + "}", new byte[24]);
        Files.createSymbolicLink(child.resolve("linked.safetensors"), directory.resolve("outside"));
        Path index = child.resolve("index.json"); Files.writeString(index, "{\"weight_map\":{\"weight\":\"linked.safetensors\"}}");
        assertThrows(IOException.class, () -> Safetensors.openIndex(index));
    }

    @Test void readsGgufMetadataAndReversesAllLogicalDimensions() throws Exception {
        var header = ggufHeader(1, 3);
        string(header, "general.architecture"); i32(header, 8); string(header, "test");
        string(header, "general.alignment"); i32(header, 4); i32(header, 32);
        string(header, "tokenizer.ggml.tokens"); i32(header, 9); i32(header, 8); i64(header, 2); string(header, "a"); string(header, "é");
        string(header, "rank4"); i32(header, 4); for (int dim : new int[]{2, 3, 4, 5}) { i64(header, dim); }
        i32(header, 0); i64(header, 0);
        Path path = gguf("rank4.gguf", header, new byte[2 * 3 * 4 * 5 * 4]);
        try (var file = Gguf.open(path)) {
            assertEquals(List.of(5L, 4L, 3L, 2L), file.tensors().get("rank4").shape());
            assertEquals(List.of("a", "é"), file.metadata().get("tokenizer.ggml.tokens"));
            assertEquals("test", file.metadata().get("general.architecture")); assertEquals(480, file.data("rank4").byteSize());
        }
    }

    @Test void preservesQuantizedGgmlBlocksWithoutLoadingGpuModules() throws Exception {
        int[] ids = {2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15};
        int[] elements = {32, 32, 32, 32, 32, 256, 256, 256, 256, 256, 256};
        int[] bytes = {18, 20, 22, 24, 34, 84, 110, 144, 176, 210, 292};
        for (int i = 0; i < ids.length; i++) {
            var header = ggufHeader(1, 0); tensor(header, "q", elements[i], 2, ids[i], 0);
            byte[] payload = new byte[bytes[i] * 2]; java.util.Arrays.fill(payload, (byte) 0xab);
            try (var file = Gguf.open(gguf("quant", header, payload))) {
                assertTrue(file.tensors().get("q").type().quantized()); assertArrayEquals(payload, file.data("q").toArray(ValueLayout.JAVA_BYTE));
            }
        }
    }

    @Test void rejectsGgufVersionTypesAlignmentAndOverlappingRanges() throws Exception {
        var badVersion = ggufHeader(0, 0).toByteArray(); badVersion[4] = 2;
        Path version = directory.resolve("version"); Files.write(version, badVersion); assertThrows(IOException.class, () -> Gguf.open(version));
        for (int type : new int[]{-1, 999}) {
            var header = ggufHeader(1, 0); tensor(header, "q", 32, 1, type, 0);
            assertThrows(IOException.class, () -> Gguf.open(gguf("type", header, new byte[128])));
        }
        var alignment = ggufHeader(0, 1); string(alignment, "general.alignment"); i32(alignment, 4); i32(alignment, 3);
        assertThrows(IOException.class, () -> Gguf.open(gguf("align", alignment, new byte[0])));
        var overlap = ggufHeader(2, 0); tensor(overlap, "a", 8, 1, 0, 0); tensor(overlap, "b", 8, 1, 0, 0);
        assertThrows(IOException.class, () -> Gguf.open(gguf("overlap", overlap, new byte[32])));
        var unaligned = ggufHeader(1, 0); tensor(unaligned, "a", 1, 1, 0, 1);
        assertThrows(IOException.class, () -> Gguf.open(gguf("unaligned", unaligned, new byte[8])));
        var incomplete = ggufHeader(1, 0); tensor(incomplete, "q", 33, 1, 2, 0);
        assertThrows(IOException.class, () -> Gguf.open(gguf("incomplete", incomplete, new byte[128])));
    }

    @Test void rejectsTruncatedDuplicateAndMalformedGgufMetadata() throws Exception {
        var duplicate = ggufHeader(0, 2);
        for (int i = 0; i < 2; i++) { string(duplicate, "key"); i32(duplicate, 8); string(duplicate, "value"); }
        assertThrows(IOException.class, () -> Gguf.open(gguf("dup", duplicate, new byte[0])));
        var utf = ggufHeader(0, 1); i64(utf, 1); utf.write(0xff); i32(utf, 8); string(utf, "x");
        assertThrows(IOException.class, () -> Gguf.open(gguf("utf", utf, new byte[0])));
        var nested = ggufHeader(0, 1); string(nested, "array"); i32(nested, 9); i32(nested, 9); i64(nested, 0);
        assertThrows(IOException.class, () -> Gguf.open(gguf("nested", nested, new byte[0])));
        var header = ggufHeader(1, 0); tensor(header, "a", 8, 1, 0, 0);
        byte[] full = Files.readAllBytes(gguf("full", header, new byte[32]));
        for (int size : new int[]{0, 8, 23, 31, full.length - 1}) {
            Path path = directory.resolve("truncated"); Files.write(path, java.util.Arrays.copyOf(full, size));
            assertThrows(IOException.class, () -> Gguf.open(path));
        }
    }

    @Test void jsonRejectsDuplicateKeysInvalidUtf8SurrogatesAndTrailingDocuments() throws Exception {
        for (String text : List.of("{\"x\":1,\"x\":2}", "{\"a\":[}", "{\"x\":\"\\ud800\"}", "{}{}", "{\"x\":9223372036854775808}")) {
            assertThrows(IOException.class, () -> JsonDocuments.parse(text.getBytes(StandardCharsets.UTF_8)));
        }
        assertThrows(IOException.class, () -> JsonDocuments.parse(new byte[]{'{', '"', 'x', '"', ':', '"', (byte) 0xff, '"', '}'}));
        assertThrows(IOException.class, () -> JsonDocuments.parse("{\"x\":1}".getBytes(StandardCharsets.UTF_16LE)));
        assertEquals("😀", JsonDocuments.parse("{\"s\":\"\\ud83d\\ude00\"}".getBytes(StandardCharsets.UTF_8)).get("s"));
        assertThrows(IOException.class, () -> JsonDocuments.parse(new byte[JsonDocuments.MAX_BYTES + 1]));
    }

    private Path safe(String name, String header, byte[] data) throws IOException {
        byte[] bytes = header.getBytes(StandardCharsets.UTF_8);
        Path path = directory.resolve(name); Files.write(path, ByteBuffer.allocate(8 + bytes.length + data.length)
                .order(ByteOrder.LITTLE_ENDIAN).putLong(bytes.length).put(bytes).put(data).array()); return path;
    }
    private static ByteArrayOutputStream ggufHeader(int tensors, int metadata) {
        var output = new ByteArrayOutputStream(); i32(output, 0x46554747); i32(output, 3); i64(output, tensors); i64(output, metadata); return output;
    }
    private Path gguf(String name, ByteArrayOutputStream header, byte[] data) throws IOException {
        while (header.size() % 32 != 0) { header.write(0); } header.writeBytes(data);
        Path path = directory.resolve(name); Files.write(path, header.toByteArray()); return path;
    }
    private static void tensor(ByteArrayOutputStream output, String name, long columns, long rows, int type, long offset) {
        string(output, name); i32(output, 2); i64(output, columns); i64(output, rows); i32(output, type); i64(output, offset);
    }
    private static void string(ByteArrayOutputStream output, String value) { byte[] bytes = value.getBytes(StandardCharsets.UTF_8); i64(output, bytes.length); output.writeBytes(bytes); }
    private static void i32(ByteArrayOutputStream output, int value) { output.writeBytes(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(value).array()); }
    private static void i64(ByteArrayOutputStream output, long value) { output.writeBytes(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(value).array()); }
}

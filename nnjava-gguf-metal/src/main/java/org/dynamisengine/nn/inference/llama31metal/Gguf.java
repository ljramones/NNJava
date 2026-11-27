package org.dynamisengine.nn.inference.llama31metal;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.Map;

import org.dynamisengine.gpu.primitives.GgufBlocks;

/** Read-only GGUF v3 reader for the FP32 vectors and GGML matrices on the supported Llama 3.1 checkpoint. */
final class Gguf implements AutoCloseable {
    private static final ValueLayout.OfInt I32 = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfShort I16 = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong I64 = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfFloat F32 = ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    /** Absolute file offset. GGUF dimension zero is contiguous (cols), and dimension one is rows. */
    public record Tensor(String name, int rank, int rows, int cols, int typeId, long offset, long byteSize) { }

    private final Arena arena;
    private final MemorySegment data;
    private final Map<String, Tensor> tensors = new LinkedHashMap<>();
    private final Map<String, Object> metadata = new LinkedHashMap<>();

    public static Gguf open(Path path) throws IOException {
        Arena arena = Arena.ofShared();
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            return new Gguf(arena, channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size(), arena));
        } catch (IOException | RuntimeException | Error failure) {
            arena.close();
            throw failure;
        }
    }

    private Gguf(Arena arena, MemorySegment data) {
        this.arena = arena;
        this.data = data;
        Cursor in = new Cursor(data);
        if (in.int32() != 0x46554747 || in.int32() != 3) {
            throw new IllegalArgumentException("Expected little-endian GGUF version 3");
        }
        long tensorCount = in.count();
        long metadataCount = in.count();
        for (long i = 0; i < metadataCount; i++) {
            String key = in.string();
            Object value = in.metadata(in.int32());
            if (metadata.putIfAbsent(key, value) != null) {
                throw new IllegalArgumentException("Duplicate GGUF key " + key);
            }
        }
        long alignment = metadata.containsKey("general.alignment") ? integer("general.alignment") : 32;
        if (alignment <= 0 || (alignment & (alignment - 1)) != 0) {
            throw new IllegalArgumentException("Invalid GGUF alignment " + alignment);
        }
        var descriptors = new ArrayList<Tensor>();
        for (long i = 0; i < tensorCount; i++) {
            String name = in.string();
            int rank = in.int32();
            if (rank != 1 && rank != 2) throw new IllegalArgumentException("Unsupported tensor rank " + rank + " for " + name);
            int cols = Math.toIntExact(in.count());
            int rows = rank == 2 ? Math.toIntExact(in.count()) : 1;
            if (cols < 1 || rows < 1) throw new IllegalArgumentException("Empty tensor " + name);
            int type = in.int32();
            long relative = in.count();
            if (relative % alignment != 0) throw new IllegalArgumentException("Misaligned tensor " + name);
            long bytes = type == 0 ? Math.multiplyExact(Math.multiplyExact((long) rows, cols), 4)
                    : GgufBlocks.bytes(GgufBlocks.Type.of(type), rows, cols);
            descriptors.add(new Tensor(name, rank, rows, cols, type, relative, bytes));
        }
        long base = Math.addExact(in.offset, alignment - 1) & -alignment;
        long end = base;
        descriptors.sort(Comparator.comparingLong(Tensor::offset));
        for (var desc : descriptors) {
            long start = Math.addExact(base, desc.offset());
            if (start < end) throw new IllegalArgumentException("Overlapping tensor " + desc.name());
            end = Math.addExact(start, desc.byteSize());
            if (end > data.byteSize()) throw new IllegalArgumentException("Truncated tensor " + desc.name());
            var entry = new Tensor(desc.name(), desc.rank(), desc.rows(), desc.cols(), desc.typeId(), start, desc.byteSize());
            if (tensors.putIfAbsent(entry.name(), entry) != null) throw new IllegalArgumentException("Duplicate tensor " + entry.name());
        }
    }

    public int tensorCount() { return tensors.size(); }

    public Tensor tensor(String name) {
        Tensor tensor = tensors.get(name);
        if (tensor == null) throw new IllegalArgumentException("Missing GGUF tensor " + name);
        return tensor;
    }

    boolean flag(String key, boolean fallback) {
        return metadata.containsKey(key) ? require(key, Boolean.class) : fallback;
    }

    String[] strings(String key) {
        var array = require(key, ArrayDescriptor.class);
        if (array.type != 8) throw new IllegalArgumentException("Expected GGUF string array " + key);
        var cursor = new Cursor(data);
        cursor.offset = array.offset;
        var result = new String[Math.toIntExact(array.size)];
        for (int i = 0; i < result.length; i++) result[i] = cursor.string();
        return result;
    }

    void checkLlamaSpecialTokens() {
        var array = require("tokenizer.ggml.tokens", ArrayDescriptor.class);
        int[] types = integers("tokenizer.ggml.token_type");
        int[] ids = {128000, 128001, 128008, 128009};
        String[] expected = {"<|begin_of_text|>", "<|end_of_text|>", "<|eom_id|>", "<|eot_id|>"};
        if (array.type != 8 || array.size != 128256 || types.length != array.size)
            throw new IllegalArgumentException("Invalid Llama 3.1 vocabulary size or type");
        var cursor = new Cursor(data);
        cursor.offset = array.offset;
        int next = 0;
        for (int i = 0; next < ids.length; i++) {
            if (i != ids[next]) {
                cursor.take(cursor.count());
            } else {
                if (types[i] != 3 || !cursor.string().equals(expected[next]))
                    throw new IllegalArgumentException("Unexpected Llama 3.1 control token "+i);
                next++;
            }
        }
    }

    int[] integers(String key) {
        var array = require(key, ArrayDescriptor.class);
        if (array.type != 5) throw new IllegalArgumentException("Expected GGUF int32 array " + key);
        return data.asSlice(array.offset, Math.multiplyExact(array.size, 4L)).toArray(I32);
    }

    public int integer(String key) {
        return Math.toIntExact(require(key, Long.class));
    }

    public double decimal(String key) {
        return require(key, Number.class).doubleValue();
    }

    public String text(String key) {
        return require(key, String.class);
    }

    private <T> T require(String key, Class<T> type) {
        Object value = metadata.get(key);
        if (!type.isInstance(value)) throw new IllegalArgumentException("Missing or invalid metadata " + key);
        return type.cast(value);
    }

    public float[] floats(String name) {
        var tensor = tensor(name);
        if (tensor.typeId() != 0) throw new IllegalArgumentException("Expected F32 tensor " + name);
        return data.asSlice(tensor.offset(), tensor.byteSize()).toArray(F32);
    }

    /** Repack the original codes and scales to DynamisGPU's layout without re-quantizing. */
    public GgufBlocks.Packed matrix(String name) {
        var tensor = tensor(name);
        return rows(tensor, 0, tensor.rows());
    }

    GgufBlocks.Packed rows(Tensor tensor, int first, int count) {
        if (first < 0 || count < 1 || count > tensor.rows() - first) {
            throw new IllegalArgumentException("Invalid row range for " + tensor.name());
        }
        var type = GgufBlocks.Type.of(tensor.typeId());
        long rowBytes = GgufBlocks.bytes(type, 1, tensor.cols());
        var source = data.asSlice(tensor.offset() + first * rowBytes, count * rowBytes);
        return GgufBlocks.fromGgml(type, source.toArray(ValueLayout.JAVA_BYTE), 0, count, tensor.cols());
    }

    /** Decode just the requested token's original GGML row. */
    public float[] embedding(int tokenId) {
        var packed = rows(tensor("token_embd.weight"), tokenId, 1);
        return packed.matrix().dequantize(packed.bytes());
    }

    @Override public void close() { arena.close(); }

    private static final class Cursor {
        private final MemorySegment data;
        private long offset;

        Cursor(MemorySegment data) { this.data = data; }

        long take(long bytes) {
            if (bytes < 0 || bytes > data.byteSize() - offset) throw new IllegalArgumentException("Truncated GGUF at " + offset);
            long start = offset;
            offset += bytes;
            return start;
        }

        int int32() { return data.get(I32, take(4)); }

        long count() {
            long n = data.get(I64, take(8));
            if (n < 0) throw new IllegalArgumentException("GGUF count or offset exceeds signed 64-bit range");
            return n;
        }

        String string() {
            int size = Math.toIntExact(count());
            return new String(data.asSlice(take(size), size).toArray(ValueLayout.JAVA_BYTE), StandardCharsets.UTF_8);
        }

        Object metadata(int type) {
            return switch (type) {
                case 0 -> (long) Byte.toUnsignedInt(data.get(ValueLayout.JAVA_BYTE, take(1)));
                case 1 -> (long) data.get(ValueLayout.JAVA_BYTE, take(1));
                case 2 -> (long) Short.toUnsignedInt(data.get(I16, take(2)));
                case 3 -> (long) data.get(I16, take(2));
                case 4 -> Integer.toUnsignedLong(int32());
                case 5 -> (long) int32();
                case 6 -> data.get(F32, take(4));
                case 7 -> data.get(ValueLayout.JAVA_BYTE, take(1)) != 0;
                case 8 -> string();
                case 9 -> {
                    int elementType = int32();
                    long n = count();
                    long start = offset;
                    if (elementType == 8) {
                        for (long i = 0; i < n; i++) take(count());
                    } else {
                        int width = switch (elementType) {
                            case 0, 1, 7 -> 1;
                            case 2, 3 -> 2;
                            case 4, 5, 6 -> 4;
                            case 10, 11, 12 -> 8;
                            default -> throw new IllegalArgumentException("Unsupported metadata array type " + elementType);
                        };
                        take(Math.multiplyExact(n, width));
                    }
                    yield new ArrayDescriptor(elementType, n, start);
                }
                case 10 -> count();
                case 11 -> data.get(I64, take(8));
                case 12 -> Double.longBitsToDouble(data.get(I64, take(8)));
                default -> throw new IllegalArgumentException("Unknown metadata type " + type);
            };
        }
    }

    private record ArrayDescriptor(int type, long size, long offset) { }
}

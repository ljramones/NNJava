package org.dynamisengine.nn.formats;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.math.BigInteger;
import java.nio.ByteOrder;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;

/**
 * Little-endian GGUF v3 metadata and mapped tensors, including original GGML quantized block bytes.
 */
public final class Gguf {
    private static final int MAX_ENTRIES = 1_000_000;
    private static final long MAX_HEADER = 64L * 1024 * 1024;
    private static final ValueLayout.OfShort SHORT = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfInt INT = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LONG = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private Gguf() {
    }

    public static TensorFile open(Path path) throws IOException {
        var file = new MappedTensorFile(path);
        try {
            var cursor = new Cursor(file.bytes);
            if (cursor.i32() != 0x46554747 || cursor.i32() != 3) {
                throw new IOException("Expected little-endian GGUF v3");
            }
            int tensorCount = cursor.count(), metadataCount = cursor.count();
            var metadata = new LinkedHashMap<String, Object>();
            for (int i = 0; i < metadataCount; i++) {
                String key = cursor.string();
                Object value = cursor.value(cursor.i32(), true);
                if (metadata.putIfAbsent(key, value) != null) {
                    throw new IOException("Duplicate GGUF metadata: " + key);
                }
            }
            long alignment = metadata.containsKey("general.alignment")
                    ? positiveInteger(metadata.get("general.alignment")) : 32;
            if (alignment > 1024 * 1024 || (alignment & (alignment - 1)) != 0) {
                throw new IOException("Invalid GGUF alignment");
            }
            var relative = new LinkedHashMap<String, TensorDescriptor>();
            for (int i = 0; i < tensorCount; i++) {
                String name = cursor.string();
                int rank = cursor.i32();
                if (rank < 1 || rank > 4) {
                    throw new IOException("GGUF rank must be between one and four");
                }
                var shape = new ArrayList<Long>();
                long elements = 1;
                for (int d = 0; d < rank; d++) {
                    long dimension = cursor.nonnegative();
                    if (dimension == 0) {
                        throw new IOException("Zero GGUF dimension");
                    }
                    shape.add(dimension);
                    elements = Math.multiplyExact(elements, dimension);
                }
                Collections.reverse(shape); // GGML ne[0] is the contiguous (last logical) dimension.
                var type = TensorType.ggml(cursor.i32());
                long offset = cursor.nonnegative();
                if (offset % alignment != 0) {
                    throw new IOException("Unaligned GGUF tensor offset");
                }
                var descriptor = new TensorDescriptor(name, type, shape, offset, type.byteSize(elements));
                if (relative.putIfAbsent(name, descriptor) != null) {
                    throw new IOException("Duplicate GGUF tensor: " + name);
                }
            }
            long base = Math.addExact(cursor.position, alignment - 1) & -alignment;
            if (base > file.bytes.byteSize()) {
                throw new IOException("Truncated GGUF alignment padding");
            }
            var tensors = new LinkedHashMap<String, TensorDescriptor>();
            for (var tensor : relative.values()) {
                tensors.put(tensor.name(), new TensorDescriptor(tensor.name(), tensor.type(), tensor.shape(),
                        Math.addExact(base, tensor.offset()), tensor.byteSize()));
            }
            file.finish(metadata, tensors, base, false);
            return file;
        } catch (IOException | RuntimeException | Error failure) {
            file.close();
            if (failure instanceof RuntimeException) {
                throw new IOException("Invalid GGUF file: " + path, failure);
            }
            throw failure;
        }
    }

    private static long positiveInteger(Object value) throws IOException {
        if (!(value instanceof Long number) || number <= 0) {
            throw new IOException("Expected positive GGUF integer");
        }
        return number;
    }

    private static final class Cursor {
        private final MemorySegment bytes;
        long position;
        private int metadataValues;

        Cursor(MemorySegment bytes) {
            this.bytes = bytes;
        }

        long take(long size) throws IOException {
            if (size < 0 || size > bytes.byteSize() - position || size > MAX_HEADER - position) {
                throw new IOException("Truncated or excessive GGUF header");
            }
            long result = position;
            position += size;
            return result;
        }

        int i32() throws IOException {
            return bytes.get(INT, take(4));
        }

        long i64() throws IOException {
            return bytes.get(LONG, take(8));
        }

        long nonnegative() throws IOException {
            long value = i64();
            if (value < 0) {
                throw new IOException("GGUF size exceeds signed 64-bit range");
            }
            return value;
        }

        int count() throws IOException {
            long value = nonnegative();
            if (value > MAX_ENTRIES) {
                throw new IOException("Excessive GGUF entry count");
            }
            return (int) value;
        }

        String string() throws IOException {
            long size = nonnegative();
            if (size > JsonDocuments.MAX_BYTES) {
                throw new IOException("Excessive GGUF string");
            }
            var buffer = bytes.asSlice(take(size), size).asByteBuffer();
            try {
                return StandardCharsets.UTF_8.newDecoder().onMalformedInput(CodingErrorAction.REPORT)
                        .onUnmappableCharacter(CodingErrorAction.REPORT).decode(buffer).toString();
            } catch (CharacterCodingException failure) {
                throw new IOException("Invalid UTF-8 in GGUF", failure);
            }
        }

        Object value(int type, boolean allowArray) throws IOException {
            if (++metadataValues > 2_000_000) {
                throw new IOException("Excessive total GGUF metadata values");
            }
            return switch (type) {
                case 0 -> (long) Byte.toUnsignedInt(bytes.get(ValueLayout.JAVA_BYTE, take(1)));
                case 1 -> (long) bytes.get(ValueLayout.JAVA_BYTE, take(1));
                case 2 -> (long) Short.toUnsignedInt(bytes.get(SHORT, take(2)));
                case 3 -> (long) bytes.get(SHORT, take(2));
                case 4 -> Integer.toUnsignedLong(i32());
                case 5 -> (long) i32();
                case 6 -> Float.intBitsToFloat(i32());
                case 7 -> {
                    byte value = bytes.get(ValueLayout.JAVA_BYTE, take(1));
                    if (value != 0 && value != 1) {
                        throw new IOException("Invalid GGUF boolean");
                    }
                    yield value != 0;
                }
                case 8 -> string();
                case 9 -> {
                    if (!allowArray) {
                        throw new IOException("Nested GGUF metadata arrays are unsupported");
                    }
                    int elementType = i32(), count = count();
                    if (elementType < 0 || elementType > 12 || elementType == 9) {
                        throw new IOException("Invalid GGUF array type");
                    }
                    var values = new ArrayList<>(count);
                    for (int i = 0; i < count; i++) {
                        values.add(value(elementType, false));
                    }
                    yield List.copyOf(values);
                }
                case 10 -> {
                    long value = i64();
                    yield value >= 0 ? value : new BigInteger(Long.toUnsignedString(value));
                }
                case 11 -> i64();
                case 12 -> Double.longBitsToDouble(i64());
                default -> throw new IOException("Unknown GGUF metadata type: " + type);
            };
        }
    }
}

package example;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.List;

import org.dynamisengine.nn.formats.Safetensors;
import org.dynamisengine.nn.formats.TensorType;

/**
 * Container parsing with no NNJava model, tensor runtime, or numerical modules on the path.
 */
public final class FormatsConsumer {
    public static void main(String[] args) throws Exception {
        var path = Files.createTempFile("nnjava-formats-", ".safetensors");
        byte[] header = "{\"weight\":{\"dtype\":\"F32\",\"shape\":[1,2],\"data_offsets\":[0,8]}}".getBytes(StandardCharsets.UTF_8);
        MemorySegment borrowed;
        try {
            Files.write(path, ByteBuffer.allocate(8 + header.length + 8).order(ByteOrder.LITTLE_ENDIAN)
                    .putLong(header.length).put(header).putFloat(3).putFloat(7).array());
            try (var file = Safetensors.open(path)) {
                var descriptor = file.tensors().get("weight");
                borrowed = file.data("weight");
                if (descriptor.type() != TensorType.F32 || !descriptor.shape().equals(List.of(1L, 2L))
                        || borrowed.get(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), 4) != 7) {
                    throw new AssertionError("Packaged container read failed");
                }
            }
            if (borrowed.scope().isAlive()) {
                throw new AssertionError("Mapped source outlived its owner");
            }
        } finally {
            Files.deleteIfExists(path);
        }
        System.out.println("Format-only JPMS consumer passed without core, model or numerical modules");
    }
}

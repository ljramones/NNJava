package org.dynamisengine.nn.io;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FilterOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.TreeMap;
import java.util.zip.CRC32;
import org.dynamisengine.nn.model.ComponentSpec;

/** Bounded, checksummed definition files. Reading does not allocate tensors or load numerical providers. */
public final class ModelDefinitions {
    private static final int MAGIC = 0x4e4e4a44;
    private static final int VERSION = 1;
    private static final int MAX_BYTES = 16 * 1024 * 1024;
    private static final int MAX_NODES = 10_000;
    private static final int MAX_ENTRIES = 4096;
    private static final int MAX_DEPTH = 64;
    private ModelDefinitions() { }

    public static void write(Path path, ComponentSpec specification) throws IOException {
        var bytes = new ByteArrayOutputStream();
        try (var out = new DataOutputStream(new LimitedOutput(bytes))) {
            out.writeInt(MAGIC); out.writeInt(VERSION);
            writeNode(out, specification, 0, new int[1]);
            out.flush();
            var crc = new CRC32(); crc.update(bytes.toByteArray());
            out.writeLong(crc.getValue());
        }
        Path absolute = path.toAbsolutePath();
        Files.createDirectories(absolute.getParent());
        Path temporary = Files.createTempFile(absolute.getParent(), ".nnjava-definition-", ".tmp");
        try {
            Files.write(temporary, bytes.toByteArray());
            try { Files.move(temporary, absolute, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING); }
            catch (AtomicMoveNotSupportedException unsupported) { Files.move(temporary, absolute, StandardCopyOption.REPLACE_EXISTING); }
        } finally { Files.deleteIfExists(temporary); }
    }

    public static ComponentSpec read(Path path) throws IOException {
        byte[] bytes;
        try (var input = Files.newInputStream(path)) { bytes = input.readNBytes(MAX_BYTES + 1); }
        if (bytes.length > MAX_BYTES || bytes.length < 16) { throw new IOException("Invalid definition file size"); }
        int payload = bytes.length - Long.BYTES;
        var crc = new CRC32(); crc.update(bytes, 0, payload);
        if (crc.getValue() != ByteBuffer.wrap(bytes, payload, Long.BYTES).getLong()) { throw new IOException("Definition checksum mismatch"); }
        try (var in = new DataInputStream(new ByteArrayInputStream(bytes, 0, payload))) {
            if (in.readInt() != MAGIC || in.readInt() != VERSION) { throw new IOException("Unsupported definition file format"); }
            ComponentSpec result = readNode(in, 0, new int[1]);
            if (in.read() != -1) { throw new IOException("Trailing definition data"); }
            return result;
        } catch (IllegalArgumentException | ArithmeticException error) { throw new IOException("Invalid model definition", error); }
    }

    private static void writeNode(DataOutputStream out, ComponentSpec spec, int depth, int[] count) throws IOException {
        limit(depth, count);
        out.writeUTF(spec.type()); out.writeInt(spec.version()); out.writeUTF(spec.name());
        entries(spec.attributes().size()); out.writeInt(spec.attributes().size());
        for (var attribute : new TreeMap<>(spec.attributes()).entrySet()) {
            out.writeUTF(attribute.getKey()); out.writeUTF(attribute.getValue());
        }
        entries(spec.children().size()); out.writeInt(spec.children().size());
        for (ComponentSpec child : spec.children()) { writeNode(out, child, depth + 1, count); }
    }

    private static ComponentSpec readNode(DataInputStream in, int depth, int[] count) throws IOException {
        limit(depth, count);
        String type = in.readUTF(); int version = in.readInt(); String name = in.readUTF();
        int attributes = entries(in.readInt()); Map<String, String> values = new LinkedHashMap<>();
        for (int i = 0; i < attributes; i++) {
            if (values.putIfAbsent(in.readUTF(), in.readUTF()) != null) { throw new IOException("Duplicate definition setting"); }
        }
        int children = entries(in.readInt()); var nodes = new ArrayList<ComponentSpec>(children);
        for (int i = 0; i < children; i++) { nodes.add(readNode(in, depth + 1, count)); }
        return new ComponentSpec(type, version, name, values, nodes);
    }

    private static void limit(int depth, int[] count) throws IOException {
        if (depth > MAX_DEPTH || ++count[0] > MAX_NODES) { throw new IOException("Definition nesting/node limit exceeded"); }
    }
    private static int entries(int count) throws IOException {
        if (count < 0 || count > MAX_ENTRIES) { throw new IOException("Definition entry limit exceeded"); }
        return count;
    }
    private static final class LimitedOutput extends FilterOutputStream {
        private int written;
        LimitedOutput(OutputStream target) { super(target); }
        private void reserve(int count) throws IOException {
            if (count > MAX_BYTES - written) { throw new IOException("Definition file exceeds size limit"); }
            written += count;
        }
        @Override public void write(int value) throws IOException { reserve(1); out.write(value); }
        @Override public void write(byte[] bytes, int offset, int length) throws IOException { reserve(length); out.write(bytes, offset, length); }
    }
}

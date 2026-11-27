package org.dynamisengine.nn.io;

import org.dynamisengine.nn.nn.Module;
import org.dynamisengine.nn.state.ComponentState;
import org.dynamisengine.nn.state.TensorSnapshot;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;

import java.io.*;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.TreeSet;
import java.util.zip.CRC32;
import java.util.zip.CheckedInputStream;
import java.util.zip.CheckedOutputStream;

/**
 * Versioned model archives and neutral component state; preserves checkpoint format v1.
 */
public final class StateArchive {
    private static final int MAGIC = 0x4e4e4a43;
    private static final int VERSION = 1;

    private StateArchive() {
    }

    public static void saveWeights(Path path, Module model) throws IOException {
        write(path, new Loaded(StateDict.capture(model), null, Map.of()));
    }

    public static void write(Path path, Loaded checkpoint) throws IOException {
        Path absolute = path.toAbsolutePath();
        Path parent = absolute.getParent();
        Files.createDirectories(parent);
        Path temporary = Files.createTempFile(parent, ".nnjava-", ".tmp");
        try {
            try (var stream = new CheckedOutputStream(Files.newOutputStream(temporary), new CRC32()); var out = new DataOutputStream(stream)) {
                out.writeInt(MAGIC);
                out.writeInt(VERSION);
                writeTensors(out, checkpoint.model.tensors());
                writeStrings(out, checkpoint.model.aliases());
                writeLongs(out, checkpoint.model.trainingState());
                writeStrings(out, checkpoint.model.configuration());
                out.writeBoolean(checkpoint.auxiliary != null);
                if (checkpoint.auxiliary != null) {
                    out.writeUTF(checkpoint.auxiliary.kind());
                    writeStrings(out, checkpoint.auxiliary.configuration());
                    writeLongs(out, checkpoint.auxiliary.counters());
                    writeTensors(out, checkpoint.auxiliary.tensors());
                }
                writeStrings(out, checkpoint.metadata);
                long crc = stream.getChecksum().getValue();
                out.writeLong(crc);
            }
            try {
                Files.move(temporary, absolute, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            } catch (AtomicMoveNotSupportedException unsupported) {
                Files.move(temporary, absolute, StandardCopyOption.REPLACE_EXISTING);
            }
        } finally {
            Files.deleteIfExists(temporary);
        }
    }

    public static Loaded read(Path path) throws IOException {
        long size = Files.size(path);
        try (var counted = new CountedInput(Files.newInputStream(path), size);
             var checked = new CheckedInputStream(counted, new CRC32()); var in = new DataInputStream(checked)) {
            if (in.readInt() != MAGIC || in.readInt() != VERSION) {
                throw new IOException("Unsupported NNJava checkpoint header/version");
            }
            StateDict model = new StateDict(readTensors(in, counted), readStrings(in), readLongs(in), readStrings(in));
            ComponentState auxiliary = null;
            if (in.readBoolean()) {
                auxiliary = new ComponentState(in.readUTF(), readStrings(in), readLongs(in), readTensors(in, counted));
            }
            Map<String, String> metadata = readStrings(in);
            long actual = checked.getChecksum().getValue();
            if (actual != in.readLong()) {
                throw new IOException("Checkpoint checksum mismatch");
            }
            if (in.read() != -1) {
                throw new IOException("Trailing checkpoint data");
            }
            return new Loaded(model, auxiliary, metadata);
        } catch (IllegalArgumentException | ArithmeticException error) {
            throw new IOException("Invalid checkpoint contents", error);
        }
    }

    private static void writeTensors(DataOutputStream out, Map<String, TensorSnapshot> tensors) throws IOException {
        out.writeInt(tensors.size());
        for (String name : new TreeSet<>(tensors.keySet())) {
            TensorSnapshot tensor = tensors.get(name);
            if (tensor.shape().rank() > 64) {
                throw new IOException("Checkpoint format supports ranks up to 64");
            }
            out.writeUTF(name);
            out.writeByte(tensor.dtype().ordinal());
            out.writeInt(tensor.shape().rank());
            for (long dimension : tensor.shape().dimensions()) {
                out.writeLong(dimension);
            }
            out.writeInt(Math.toIntExact(tensor.shape().elements()));
            switch (tensor.dtype()) {
                case FP32 -> {
                    for (float value : (float[]) tensor.valuesCopy()) {
                        out.writeFloat(value);
                    }
                }
                case INT32 -> {
                    for (int value : (int[]) tensor.valuesCopy()) {
                        out.writeInt(value);
                    }
                }
                case BOOL -> {
                    for (boolean value : (boolean[]) tensor.valuesCopy()) {
                        out.writeBoolean(value);
                    }
                }
                case FP16, BF16 -> {
                    for (short value : tensor.shorts()) {
                        out.writeShort(value);
                    }
                }
            }
        }
    }

    private static Map<String, TensorSnapshot> readTensors(DataInputStream in, CountedInput counted) throws IOException {
        int count = count(in);
        Map<String, TensorSnapshot> result = new LinkedHashMap<>();
        for (int i = 0; i < count; i++) {
            String name = in.readUTF();
            int type = in.readUnsignedByte();
            int rank = in.readInt();
            if (type >= DType.values().length || rank < 0 || rank > 64) {
                throw new IOException("Invalid tensor type/rank");
            }
            long[] dimensions = new long[rank];
            for (int axis = 0; axis < rank; axis++) {
                dimensions[axis] = in.readLong();
            }
            Shape shape = Shape.of(dimensions);
            int n = count(in);
            DType dtype = DType.values()[type];
            if (shape.elements() != n || Math.multiplyExact((long) n, dtype.bytes()) > counted.remaining() - Long.BYTES) {
                throw new IOException("Invalid/truncated tensor payload");
            }
            TensorSnapshot tensor;
            switch (dtype) {
                case FP32 -> {
                    float[] data = new float[n];
                    for (int j = 0; j < n; j++) {
                        data[j] = in.readFloat();
                    }
                    tensor = new TensorSnapshot(shape, data);
                }
                case INT32 -> {
                    int[] data = new int[n];
                    for (int j = 0; j < n; j++) {
                        data[j] = in.readInt();
                    }
                    tensor = new TensorSnapshot(shape, data);
                }
                case BOOL -> {
                    boolean[] data = new boolean[n];
                    for (int j = 0; j < n; j++) {
                        data[j] = in.readBoolean();
                    }
                    tensor = new TensorSnapshot(shape, data);
                }
                case FP16, BF16 -> {
                    short[] data = new short[n];
                    for (int j = 0; j < n; j++) {
                        data[j] = in.readShort();
                    }
                    tensor = new TensorSnapshot(shape, dtype, data);
                }
                default -> throw new IOException("Unknown dtype");
            }
            if (result.putIfAbsent(name, tensor) != null) {
                throw new IOException("Duplicate tensor name");
            }
        }
        return result;
    }

    private static int count(DataInputStream in) throws IOException {
        int n = in.readInt();
        if (n < 0) {
            throw new IOException("Negative checkpoint count");
        }
        return n;
    }

    private static void writeStrings(DataOutputStream out, Map<String, String> values) throws IOException {
        out.writeInt(values.size());
        for (String key : new TreeSet<>(values.keySet())) {
            out.writeUTF(key);
            out.writeUTF(values.get(key));
        }
    }

    private static Map<String, String> readStrings(DataInputStream in) throws IOException {
        int count = count(in);
        Map<String, String> result = new LinkedHashMap<>();
        for (int i = 0; i < count; i++) {
            if (result.putIfAbsent(in.readUTF(), in.readUTF()) != null) {
                throw new IOException("Duplicate metadata key");
            }
        }
        return result;
    }

    private static void writeLongs(DataOutputStream out, Map<String, Long> values) throws IOException {
        out.writeInt(values.size());
        for (String key : new TreeSet<>(values.keySet())) {
            out.writeUTF(key);
            out.writeLong(values.get(key));
        }
    }

    private static Map<String, Long> readLongs(DataInputStream in) throws IOException {
        int count = count(in);
        Map<String, Long> result = new LinkedHashMap<>();
        for (int i = 0; i < count; i++) {
            if (result.putIfAbsent(in.readUTF(), in.readLong()) != null) {
                throw new IOException("Duplicate counter key");
            }
        }
        return result;
    }

    public record Loaded(StateDict model, ComponentState auxiliary, Map<String, String> metadata) {
        public Loaded {
            metadata = Map.copyOf(metadata);
        }

        public void restoreModel(Module target) {
            model.applyTo(target);
        }
    }

    private static final class CountedInput extends FilterInputStream {
        private final long size;
        private long count;

        CountedInput(InputStream input, long size) {
            super(input);
            this.size = size;
        }

        long remaining() {
            return size - count;
        }

        @Override
        public int read() throws IOException {
            int value = in.read();
            if (value >= 0) {
                count++;
            }
            return value;
        }

        @Override
        public int read(byte[] buffer, int offset, int length) throws IOException {
            int n = in.read(buffer, offset, length);
            if (n > 0) {
                count += n;
            }
            return n;
        }
    }
}

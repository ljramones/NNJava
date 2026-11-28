package org.dynamisengine.nn.formats;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.Map;

final class MappedTensorFile implements TensorFile {
    final Arena arena = Arena.ofShared();
    final MemorySegment bytes;
    private Map<String, Object> metadata;
    private Map<String, TensorDescriptor> tensors;
    private boolean closed;

    MappedTensorFile(Path path) throws IOException {
        try (var channel = FileChannel.open(path, StandardOpenOption.READ)) {
            if (channel.size() < 8) { throw new IOException("Truncated tensor file: " + path); }
            bytes = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size(), arena);
        } catch (IOException | RuntimeException | Error failure) { arena.close(); throw failure; }
    }

    void finish(Map<String, Object> metadata, Map<String, TensorDescriptor> tensors, long base, boolean contiguous) throws IOException {
        long end = base;
        var sorted = tensors.values().stream().sorted(Comparator.comparingLong(TensorDescriptor::offset)
                .thenComparingLong(TensorDescriptor::byteSize)).toList();
        for (var tensor : sorted) {
            if (tensor.offset() < end || tensor.offset() > bytes.byteSize() || tensor.byteSize() > bytes.byteSize() - tensor.offset()) {
                throw new IOException("Overlapping or out-of-range tensor: " + tensor.name());
            }
            if (contiguous && tensor.offset() != end) { throw new IOException("Unused safetensors data bytes"); }
            end = tensor.offset() + tensor.byteSize();
        }
        if (contiguous && end != bytes.byteSize()) { throw new IOException("Trailing safetensors data bytes"); }
        this.metadata = Collections.unmodifiableMap(new LinkedHashMap<>(metadata));
        this.tensors = Collections.unmodifiableMap(new LinkedHashMap<>(tensors));
    }

    private void requireOpen() { if (closed) { throw new IllegalStateException("Tensor file is closed"); } }
    @Override public Map<String, Object> metadata() { requireOpen(); return metadata; }
    @Override public Map<String, TensorDescriptor> tensors() { requireOpen(); return tensors; }
    @Override public MemorySegment data(String name) {
        requireOpen(); var descriptor = tensors.get(name);
        if (descriptor == null) { throw new IllegalArgumentException("Unknown tensor: " + name); }
        return bytes.asSlice(descriptor.offset(), descriptor.byteSize());
    }
    @Override public void close() { if (!closed) { closed = true; arena.close(); } }
}

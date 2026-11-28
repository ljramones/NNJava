package org.dynamisengine.nn.formats;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.*;

/**
 * Checked safetensors files and Hugging Face shard indexes. No numerical backend is loaded.
 */
public final class Safetensors {
    private static final ValueLayout.OfLong LE_LONG = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private Safetensors() {
    }

    public static TensorFile open(Path path) throws IOException {
        var file = new MappedTensorFile(path);
        try {
            long length = file.bytes.get(LE_LONG, 0);
            if (length < 2 || length > JsonDocuments.MAX_BYTES || length > file.bytes.byteSize() - 8) {
                throw new IOException("Invalid safetensors header length");
            }
            if (file.bytes.get(ValueLayout.JAVA_BYTE, 8) != '{') {
                throw new IOException("Safetensors header must start with '{'");
            }
            var header = JsonDocuments.parse(file.bytes.asSlice(8, length).toArray(ValueLayout.JAVA_BYTE));
            var metadata = new LinkedHashMap<String, Object>();
            if (header.containsKey("__metadata__")) {
                for (var entry : JsonDocuments.object(header.get("__metadata__"), "__metadata__").entrySet()) {
                    metadata.put(entry.getKey(), JsonDocuments.string(entry.getValue(), entry.getKey()));
                }
            }
            long base = 8 + length;
            var tensors = new LinkedHashMap<String, TensorDescriptor>();
            for (var entry : header.entrySet()) {
                if (entry.getKey().equals("__metadata__")) {
                    continue;
                }
                var fields = JsonDocuments.object(entry.getValue(), entry.getKey());
                if (!fields.keySet().equals(Set.of("dtype", "shape", "data_offsets"))) {
                    throw new IOException("Invalid tensor fields: " + entry.getKey());
                }
                var type = TensorType.valueOf(JsonDocuments.string(fields.get("dtype"), "dtype"));
                if (type.quantized()) {
                    throw new IOException("GGML block types are not safetensors dtypes");
                }
                var shape = new java.util.ArrayList<Long>();
                for (var dim : JsonDocuments.array(fields.get("shape"), "shape")) {
                    shape.add(JsonDocuments.integer(dim, "dimension"));
                }
                var offsets = JsonDocuments.array(fields.get("data_offsets"), "data_offsets");
                if (offsets.size() != 2) {
                    throw new IOException("Expected two data offsets");
                }
                long start = JsonDocuments.integer(offsets.get(0), "start"), end = JsonDocuments.integer(offsets.get(1), "end");
                if (start < 0 || end < start) {
                    throw new IOException("Invalid tensor range");
                }
                tensors.put(entry.getKey(), new TensorDescriptor(entry.getKey(), type, shape, Math.addExact(base, start), end - start));
            }
            file.finish(metadata, tensors, base, true);
            return file;
        } catch (IOException | RuntimeException | Error failure) {
            file.close();
            if (failure instanceof RuntimeException) {
                throw new IOException("Invalid safetensors file: " + path, failure);
            }
            throw failure;
        }
    }

    /**
     * Opens only local sibling shards; rejects traversal, escaping symlinks, missing and unindexed tensors.
     */
    public static TensorFile openIndex(Path index) throws IOException {
        var document = JsonDocuments.read(index);
        var mapping = JsonDocuments.object(document.get("weight_map"), "weight_map");
        if (mapping.isEmpty()) {
            throw new IOException("Empty shard index");
        }
        var root = index.toAbsolutePath().getParent().toRealPath();
        var shards = new LinkedHashMap<String, TensorFile>();
        var owners = new LinkedHashMap<String, TensorFile>();
        var tensors = new LinkedHashMap<String, TensorDescriptor>();
        try {
            for (var entry : mapping.entrySet()) {
                String name = JsonDocuments.string(entry.getValue(), "shard filename");
                Path relative = Path.of(name);
                if (relative.isAbsolute() || relative.getNameCount() != 1 || name.equals(".") || name.equals("..") || name.contains("\\")) {
                    throw new IOException("Shard must be a sibling filename: " + name);
                }
                var shard = shards.get(name);
                if (shard == null) {
                    if (shards.size() >= 1024) {
                        throw new IOException("Too many shards");
                    }
                    var actual = root.resolve(relative).toRealPath();
                    if (!actual.getParent().equals(root)) {
                        throw new IOException("Shard escapes index directory");
                    }
                    shard = open(actual);
                    shards.put(name, shard);
                }
                var descriptor = shard.tensors().get(entry.getKey());
                if (descriptor == null) {
                    throw new IOException("Index tensor absent from shard: " + entry.getKey());
                }
                owners.put(entry.getKey(), shard);
                tensors.put(entry.getKey(), descriptor);
            }
            var seen = new LinkedHashSet<String>();
            for (var shard : shards.values()) {
                for (String name : shard.tensors().keySet()) {
                    if (owners.get(name) != shard || !seen.add(name)) {
                        throw new IOException("Unindexed or duplicated shard tensor: " + name);
                    }
                }
            }
            var metadata = document.containsKey("metadata") ? JsonDocuments.object(document.get("metadata"), "metadata") : Map.<String, Object>of();
            return new ShardedFile(List.copyOf(shards.values()), Map.copyOf(owners),
                    Collections.unmodifiableMap(tensors), metadata);
        } catch (IOException | RuntimeException | Error failure) {
            shards.values().forEach(TensorFile::close);
            if (failure instanceof RuntimeException) {
                throw new IOException("Invalid safetensors index: " + index, failure);
            }
            throw failure;
        }
    }

    private static final class ShardedFile implements TensorFile {
        private final List<TensorFile> shards;
        private final Map<String, TensorFile> owners;
        private final Map<String, TensorDescriptor> tensors;
        private final Map<String, Object> metadata;
        private boolean closed;

        ShardedFile(List<TensorFile> shards, Map<String, TensorFile> owners, Map<String, TensorDescriptor> tensors, Map<String, Object> metadata) {
            this.shards = shards;
            this.owners = owners;
            this.tensors = tensors;
            this.metadata = metadata;
        }

        private void requireOpen() {
            if (closed) {
                throw new IllegalStateException("Tensor file is closed");
            }
        }

        public Map<String, Object> metadata() {
            requireOpen();
            return metadata;
        }

        public Map<String, TensorDescriptor> tensors() {
            requireOpen();
            return tensors;
        }

        public MemorySegment data(String name) {
            requireOpen();
            var owner = owners.get(name);
            if (owner == null) {
                throw new IllegalArgumentException("Unknown tensor: " + name);
            }
            return owner.data(name);
        }

        public void close() {
            if (!closed) {
                closed = true;
                shards.forEach(TensorFile::close);
            }
        }
    }
}

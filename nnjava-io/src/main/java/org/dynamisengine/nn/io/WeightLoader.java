package org.dynamisengine.nn.io;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import org.dynamisengine.nn.formats.TensorDescriptor;
import org.dynamisengine.nn.formats.TensorFile;
import org.dynamisengine.nn.formats.TensorType;
import org.dynamisengine.nn.nn.Module;
import org.dynamisengine.nn.nn.Parameter;
import org.dynamisengine.nn.state.TensorSnapshot;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;

/** Explicit, checked pretrained mappings. Owns neither model nor source; stages one tensor at a time. */
public final class WeightLoader {
    private static final ValueLayout.OfInt INT = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfShort SHORT = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private record Mapping(String source, String target, boolean transpose) { }
    private record Target(Mapping mapping, TensorDescriptor descriptor, Parameter parameter, Shape shape, DType dtype) { }
    private record Plan(List<Target> targets, Report report) { }
    /** Target aliases share storage; only one assignment is made to each unique parameter. */
    public record Report(Map<String, String> sourceToTarget, Map<String, String> aliases, Set<String> ignoredSources, long sourceBytes) {
        public Report {
            sourceToTarget = Collections.unmodifiableMap(new LinkedHashMap<>(sourceToTarget));
            aliases = Map.copyOf(aliases); ignoredSources = Set.copyOf(ignoredSources);
        }
        public int uniqueParameters() { return sourceToTarget.size(); }
    }
    private final List<Mapping> mappings;
    private final Set<String> ignored;
    private final boolean allSources, floatingToFp32;
    private final long maxTensorBytes;

    private WeightLoader(Builder builder) {
        mappings = List.copyOf(builder.mappings); ignored = Set.copyOf(builder.ignored);
        allSources = builder.allSources; floatingToFp32 = builder.floatingToFp32; maxTensorBytes = builder.maxTensorBytes;
    }
    public static Builder builder() { return new Builder(); }

    /** Reads one unquantized tensor into an owned, backend-neutral host snapshot, preserving its dtype. */
    public static TensorSnapshot readTensor(TensorFile source, String name) {
        return readTensor(source, name, 256L * 1024 * 1024);
    }

    public static TensorSnapshot readTensor(TensorFile source, String name, long maxTensorBytes) {
        var descriptor = source.tensors().get(name);
        if (descriptor == null) { throw new IllegalArgumentException("Unknown source tensor: " + name); }
        if (maxTensorBytes <= 0 || descriptor.byteSize() > maxTensorBytes || descriptor.elements() > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Tensor exceeds import budget: " + name);
        }
        var data = source.data(name);
        if (data.byteSize() != descriptor.byteSize()) { throw new IllegalArgumentException("Source data size changed"); }
        return snapshot(data, new Target(new Mapping(name, name, false), descriptor, null,
                Shape.of(descriptor.dimensions()), dtype(descriptor.type())));
    }

    /** Checks complete target coverage, ties, shapes, storage, budgets and source names without mutating the model. */
    public Report validate(TensorFile source, Module model) { return plan(source, model).report; }

    /**
     * Performs complete structural validation before writing. Backend/allocation failures during assignment can
     * leave a partially loaded model; discard that model. Do not modify/close source or use model concurrently.
     */
    public Report load(TensorFile source, Module model) {
        var plan = plan(source, model);
        for (var target : plan.targets) {
            var snapshot = snapshot(source.data(target.mapping.source), target);
            try (var tensor = target.parameter.value()) {
                snapshot.assignTo(tensor); tensor.context().synchronize(); tensor.clearGradient();
            }
        }
        return plan.report;
    }

    private Plan plan(TensorFile source, Module model) {
        Objects.requireNonNull(source); Objects.requireNonNull(model);
        var descriptors = source.tensors(); var parameters = model.namedParameters();
        var parameterIds = new LinkedHashMap<String, Long>(); var canonical = new LinkedHashMap<Long, String>();
        var aliases = new LinkedHashMap<String, String>();
        for (var entry : parameters.entrySet()) {
            try (var tensor = entry.getValue().value()) {
                if (tensor.context().activeTape() != null) { throw new IllegalStateException("Cannot load weights during gradient recording"); }
                parameterIds.put(entry.getKey(), tensor.id());
                String first = canonical.putIfAbsent(tensor.id(), entry.getKey());
                if (first != null) { aliases.put(entry.getKey(), first); }
            }
        }
        if (!descriptors.keySet().containsAll(ignored)) { throw new IllegalArgumentException("Ignored source tensor is absent"); }
        var byId = new HashMap<Long, Mapping>(); var targets = new ArrayList<Target>();
        var used = new LinkedHashSet<String>(); var report = new LinkedHashMap<String, String>(); long bytes = 0;
        for (var mapping : mappings) {
            var descriptor = descriptors.get(mapping.source); var parameter = parameters.get(mapping.target);
            if (descriptor == null || parameter == null || ignored.contains(mapping.source)) {
                throw new IllegalArgumentException("Unknown or ignored weight mapping: " + mapping.source + " -> " + mapping.target);
            }
            long id = parameterIds.get(mapping.target);
            if (byId.putIfAbsent(id, mapping) != null) { throw new IllegalArgumentException("Map tied target storage only once: " + mapping.target); }
            if (!used.add(mapping.source)) { throw new IllegalArgumentException("Source tensor mapped more than once: " + mapping.source); }
            if (descriptor.type().quantized()) { throw new UnsupportedOperationException("Quantized blocks require a numerical backend importer: " + mapping.source); }
            Shape shape = Shape.of(descriptor.dimensions());
            if (mapping.transpose) {
                if (shape.rank() != 2) { throw new IllegalArgumentException("Transpose mapping requires a matrix"); }
                shape = Shape.of(shape.dimension(1), shape.dimension(0));
            }
            try (var tensor = parameter.value()) {
                DType sourceType = dtype(descriptor.type());
                boolean expand = floatingToFp32 && sourceType.isLowPrecision() && tensor.dtype() == DType.FP32;
                if (!tensor.shape().equals(shape) || (tensor.dtype() != sourceType && !expand)) {
                    throw new IllegalArgumentException("Weight shape/dtype mismatch: " + mapping.source + " -> " + mapping.target);
                }
                long stagingBytes = Math.multiplyExact(descriptor.elements(), tensor.dtype().bytes());
                if (descriptor.byteSize() > maxTensorBytes || stagingBytes > maxTensorBytes || descriptor.elements() > Integer.MAX_VALUE) {
                    throw new IllegalArgumentException("Tensor exceeds import budget: " + mapping.source);
                }
                var segment = source.data(mapping.source);
                if (segment.byteSize() != descriptor.byteSize()) { throw new IllegalArgumentException("Source data size changed"); }
                if (sourceType == DType.BOOL) {
                    for (long i = 0; i < segment.byteSize(); i++) {
                        byte value = segment.get(ValueLayout.JAVA_BYTE, i);
                        if (value != 0 && value != 1) { throw new IllegalArgumentException("Invalid BOOL storage: " + mapping.source); }
                    }
                }
                targets.add(new Target(mapping, descriptor, parameter, shape, tensor.dtype()));
            }
            report.put(mapping.source, mapping.target); bytes = Math.addExact(bytes, descriptor.byteSize());
        }
        if (byId.size() != canonical.size()) { throw new IllegalArgumentException("Every unique model parameter must have a weight mapping"); }
        var unused = new LinkedHashSet<>(descriptors.keySet()); unused.removeAll(used); unused.removeAll(ignored);
        if (allSources && !unused.isEmpty()) { throw new IllegalArgumentException("Unmapped source tensors: " + unused); }
        unused.addAll(ignored);
        return new Plan(List.copyOf(targets), new Report(report, aliases, unused, bytes));
    }

    private static DType dtype(TensorType type) {
        return switch (type) {
            case F32 -> DType.FP32; case F16 -> DType.FP16; case BF16 -> DType.BF16;
            case I32 -> DType.INT32; case BOOL -> DType.BOOL;
            default -> throw new UnsupportedOperationException("Unsupported import type: " + type);
        };
    }

    private static TensorSnapshot snapshot(MemorySegment segment, Target target) {
        int count = Math.toIntExact(target.descriptor.elements());
        int columns = target.mapping.transpose ? Math.toIntExact(target.descriptor.shape().getLast()) : 0;
        int rows = target.mapping.transpose ? Math.toIntExact(target.descriptor.shape().getFirst()) : 0;
        // Layout unpacking and IEEE storage expansion only; model computation remains in the selected backend.
        return switch (target.dtype) {
            case FP32 -> {
                float[] data = new float[count];
                for (int i = 0; i < count; i++) {
                    float value = switch (target.descriptor.type()) {
                        case F32 -> Float.intBitsToFloat(segment.get(INT, (long) i * 4));
                        case F16 -> Float.float16ToFloat(segment.get(SHORT, (long) i * 2));
                        case BF16 -> Float.intBitsToFloat(Short.toUnsignedInt(segment.get(SHORT, (long) i * 2)) << 16);
                        default -> throw new IllegalStateException("Preflight dtype mismatch");
                    };
                    data[index(i, target.mapping.transpose, rows, columns)] = value;
                }
                yield new TensorSnapshot(target.shape, data);
            }
            case INT32 -> {
                int[] data = new int[count];
                for (int i = 0; i < count; i++) { data[index(i, target.mapping.transpose, rows, columns)] = segment.get(INT, (long) i * 4); }
                yield new TensorSnapshot(target.shape, data);
            }
            case BOOL -> {
                boolean[] data = new boolean[count];
                for (int i = 0; i < count; i++) {
                    byte value = segment.get(ValueLayout.JAVA_BYTE, i);
                    if (value != 0 && value != 1) { throw new IllegalArgumentException("Invalid BOOL storage"); }
                    data[index(i, target.mapping.transpose, rows, columns)] = value != 0;
                }
                yield new TensorSnapshot(target.shape, data);
            }
            case FP16, BF16 -> {
                short[] data = new short[count];
                for (int i = 0; i < count; i++) { data[index(i, target.mapping.transpose, rows, columns)] = segment.get(SHORT, (long) i * 2); }
                yield new TensorSnapshot(target.shape, target.dtype, data);
            }
        };
    }
    private static int index(int i, boolean transpose, int rows, int columns) { return transpose ? (i % columns) * rows + i / columns : i; }

    public static final class Builder {
        private final List<Mapping> mappings = new ArrayList<>();
        private final Set<String> ignored = new LinkedHashSet<>();
        private boolean allSources = true, floatingToFp32;
        private long maxTensorBytes = 256L * 1024 * 1024;
        private Builder() { }
        public Builder map(String source, String target) { mappings.add(new Mapping(Objects.requireNonNull(source), Objects.requireNonNull(target), false)); return this; }
        public Builder transpose(String source, String target) { mappings.add(new Mapping(Objects.requireNonNull(source), Objects.requireNonNull(target), true)); return this; }
        public Builder ignoreSource(String name) { ignored.add(Objects.requireNonNull(name)); return this; }
        public Builder requireAllSources(boolean value) { allSources = value; return this; }
        public Builder expandFloatingPointToFp32(boolean value) { floatingToFp32 = value; return this; }
        /** Maximum source bytes AND expanded payload bytes per tensor; temporary copies add a small constant multiple. */
        public Builder maxTensorBytes(long value) { if (value <= 0) { throw new IllegalArgumentException("Positive byte budget required"); } maxTensorBytes = value; return this; }
        public WeightLoader build() { return new WeightLoader(this); }
    }
}

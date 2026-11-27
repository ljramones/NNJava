package org.dynamisengine.nn.nn;

import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.model.ComponentSpec;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;

/** Shared definition validation, stable component seeds, and checkpoint configuration. */
public final class ModelBuild {
    private ModelBuild() { }

    public static long seed(long rootSeed, String path) {
        long value = 0xcbf29ce484222325L ^ rootSeed;
        for (byte b : path.getBytes(StandardCharsets.UTF_8)) { value = (value ^ (b & 255)) * 0x100000001b3L; }
        value = (value ^ (value >>> 30)) * 0xbf58476d1ce4e5b9L;
        value = (value ^ (value >>> 27)) * 0x94d049bb133111ebL;
        return value ^ (value >>> 31);
    }

    public static void name(String name) {
        if (name == null || name.isBlank() || name.contains(".") || name.startsWith("$")) {
            throw new IllegalArgumentException("Invalid component name: " + name);
        }
    }

    public static Shape sampleShape(Shape shape) {
        if (shape.rank() == 0) { throw new IllegalArgumentException("A sample shape needs at least one axis"); }
        for (long dimension : shape.dimensions()) {
            if (dimension <= 0) { throw new IllegalArgumentException("Sample dimensions must be positive"); }
        }
        return shape;
    }

    public static void context(ExecutionContext context) {
        context.ensureOpen();
        if (context.activeTape() != null) { throw new IllegalStateException("Construct models outside gradient recording"); }
    }

    public static void require(ExecutionContext context, String path, DType dtype, Operation... operations) {
        for (Operation operation : operations) {
            if (!context.supports(operation, dtype)) {
                throw new UnsupportedOperationException(path + ": " + context.device() + " lacks " + operation + "/" + dtype);
            }
        }
    }

    public static void parameter(ExecutionContext context, String path, Shape shape, boolean normal) {
        if (shape.elements() > context.maxElements(DType.FP32)) {
            throw new IllegalArgumentException(path + ": parameter shape exceeds provider allocation limit: " + shape);
        }
        if (normal && !context.supportsNormalInitialization()) {
            throw new UnsupportedOperationException(path + ": " + context.device() + " lacks normal initialization");
        }
    }

    public static Map<String, String> configuration(ComponentSpec spec) {
        Map<String, String> result = new LinkedHashMap<>();
        flatten(spec, "definition", result);
        return Map.copyOf(result);
    }

    private static void flatten(ComponentSpec spec, String prefix, Map<String, String> result) {
        result.put(prefix + ".type", spec.type());
        result.put(prefix + ".schema", Integer.toString(spec.version()));
        result.put(prefix + ".name", spec.name());
        spec.attributes().forEach((key, value) -> result.put(prefix + ".setting." + key, value));
        for (int i = 0; i < spec.children().size(); i++) { flatten(spec.children().get(i), prefix + ".child." + i, result); }
    }
}

package org.dynamisengine.nn.model;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/** Immutable, versioned configuration tree. Contains no tensors or runtime handles. */
public record ComponentSpec(String type, int version, String name,
                            Map<String, String> attributes, List<ComponentSpec> children) {
    public ComponentSpec {
        if (type == null || type.isBlank() || version <= 0 || name == null || name.isBlank()) {
            throw new IllegalArgumentException("A component needs a type, positive schema version, and name");
        }
        attributes = Map.copyOf(attributes);
        children = List.copyOf(children);
    }

    public void require(String expectedType, int expectedVersion, String... keys) {
        if (!type.equals(expectedType) || version != expectedVersion) {
            throw new IllegalArgumentException(name + ": unsupported component " + type + " schema " + version);
        }
        if (!attributes.keySet().equals(Set.of(keys))) {
            throw new IllegalArgumentException(name + ": unexpected or missing settings for " + type);
        }
    }

    public String attribute(String key) {
        return Objects.requireNonNull(attributes.get(key), name + ": missing setting " + key);
    }

    public long longValue(String key) {
        try { return Long.parseLong(attribute(key)); }
        catch (NumberFormatException error) { throw new IllegalArgumentException(name + ": invalid integer " + key, error); }
    }

    public int intValue(String key) { return Math.toIntExact(longValue(key)); }
    public float floatValue(String key) { return Float.parseFloat(attribute(key)); }
    public double doubleValue(String key) { return Double.parseDouble(attribute(key)); }
    public boolean booleanValue(String key) {
        return switch (attribute(key)) {
            case "true" -> true;
            case "false" -> false;
            default -> throw new IllegalArgumentException(name + ": invalid boolean " + key);
        };
    }
}

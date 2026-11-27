package org.dynamisengine.nn.backend;

import org.dynamisengine.nn.tensor.Shape;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Immutable typed operator attributes, validated by each operation's contract.
 */
public final class KernelAttributes {
    public static final KernelAttributes EMPTY = new KernelAttributes(Map.of());
    private final Map<String, Object> values;

    private KernelAttributes(Map<String, Object> values) {
        this.values = Map.copyOf(values);
    }

    public static KernelAttributes of(String name, Object value) {
        return EMPTY.and(name, value);
    }

    public KernelAttributes and(String name, Object value) {
        if (value instanceof List<?> list) {
            if (list.stream().anyMatch(item -> !(item instanceof Integer))) {
                throw new IllegalArgumentException("Axis attributes require immutable integer values");
            }
            value = List.copyOf(list);
        } else if (value instanceof Float || value instanceof Double) {
            value = ((Number) value).doubleValue();
        } else if (value instanceof Byte || value instanceof Short || value instanceof Integer || value instanceof Long) {
            value = ((Number) value).longValue();
        } else if (!(value instanceof Boolean || value instanceof Shape || value instanceof String)) {
            throw new IllegalArgumentException("Unsupported kernel attribute type");
        }
        Map<String, Object> copy = new LinkedHashMap<>(values);
        copy.put(name, value);
        return new KernelAttributes(copy);
    }

    public long integer(String name) {
        Object value = required(name);
        if (!(value instanceof Long)) {
            throw new IllegalArgumentException("Expected integer attribute " + name);
        }
        return (Long) value;
    }

    public int intValue(String name) {
        return Math.toIntExact(integer(name));
    }

    public double number(String name) {
        return ((Number) required(name)).doubleValue();
    }

    public boolean flag(String name) {
        return (Boolean) required(name);
    }

    public String text(String name) {
        return (String) required(name);
    }

    public Shape shape(String name) {
        return (Shape) required(name);
    }

    public boolean has(String name) {
        return values.containsKey(name);
    }

    public int[] axes() {
        List<?> list = (List<?>) required("axes");
        return list.stream().mapToInt(value -> ((Number) value).intValue()).toArray();
    }

    private Object required(String name) {
        if (!values.containsKey(name)) {
            throw new IllegalArgumentException("Missing kernel attribute " + name);
        }
        return values.get(name);
    }
}

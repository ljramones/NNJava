package org.dynamisengine.nn.formats;

import com.fasterxml.jackson.core.*;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * Strict, bounded JSON trees. Integers are Long, decimals BigDecimal; returned trees are immutable.
 */
public final class JsonDocuments {
    public static final int MAX_BYTES = 16 * 1024 * 1024;
    private static final JsonFactory FACTORY = JsonFactory.builder()
            .enable(StreamReadFeature.STRICT_DUPLICATE_DETECTION)
            .streamReadConstraints(StreamReadConstraints.builder().maxNestingDepth(64)
                    .maxStringLength(MAX_BYTES).maxNumberLength(128).build()).build();

    private JsonDocuments() {
    }

    public static Map<String, Object> read(Path path) throws IOException {
        try (var input = Files.newInputStream(path)) {
            byte[] bytes = input.readNBytes(MAX_BYTES + 1);
            if (bytes.length > MAX_BYTES) {
                throw new IOException("JSON document exceeds 16 MiB: " + path);
            }
            return parse(bytes);
        }
    }

    public static Map<String, Object> parse(byte[] bytes) throws IOException {
        if (bytes.length > MAX_BYTES) {
            throw new IOException("JSON document exceeds 16 MiB");
        }
        // Jackson accepts escaped unpaired surrogates; reject those explicitly in all strings below.
        // Decode first: Jackson's byte parser also auto-detects UTF-16/32, which these assets do not permit.
        String document = StandardCharsets.UTF_8.newDecoder().decode(ByteBuffer.wrap(bytes)).toString();
        try (var parser = FACTORY.createParser(document)) {
            if (parser.nextToken() != JsonToken.START_OBJECT) {
                throw new IOException("Expected JSON object");
            }
            var result = object(parser);
            if (parser.nextToken() != null) {
                throw new IOException("Trailing JSON content");
            }
            return result;
        }
    }

    private static Map<String, Object> object(JsonParser parser) throws IOException {
        var result = new LinkedHashMap<String, Object>();
        while (parser.nextToken() != JsonToken.END_OBJECT) {
            if (parser.currentToken() != JsonToken.FIELD_NAME) {
                throw new IOException("Expected JSON field");
            }
            String key = scalarText(parser.currentName());
            parser.nextToken();
            result.put(key, value(parser));
        }
        return Collections.unmodifiableMap(result);
    }

    private static Object value(JsonParser parser) throws IOException {
        if (parser.currentToken() == null) {
            throw new IOException("Truncated JSON");
        }
        return switch (parser.currentToken()) {
            case START_OBJECT -> object(parser);
            case START_ARRAY -> {
                var result = new ArrayList<>();
                while (parser.nextToken() != JsonToken.END_ARRAY) {
                    result.add(value(parser));
                }
                yield Collections.unmodifiableList(result);
            }
            case VALUE_STRING -> scalarText(parser.getText());
            case VALUE_NUMBER_INT -> parser.getLongValue();
            case VALUE_NUMBER_FLOAT -> parser.getDecimalValue();
            case VALUE_TRUE -> true;
            case VALUE_FALSE -> false;
            case VALUE_NULL -> null;
            default -> throw new IOException("Unexpected JSON token: " + parser.currentToken());
        };
    }

    private static String scalarText(String text) throws IOException {
        for (int i = 0; i < text.length(); i++) {
            char ch = text.charAt(i);
            if (Character.isHighSurrogate(ch)) {
                if (++i == text.length() || !Character.isLowSurrogate(text.charAt(i))) {
                    throw new IOException("Unpaired surrogate in JSON");
                }
            } else if (Character.isLowSurrogate(ch)) {
                throw new IOException("Unpaired surrogate in JSON");
            }
        }
        return text;
    }

    public static Map<String, Object> object(Object value, String label) throws IOException {
        if (!(value instanceof Map<?, ?> map)) {
            throw new IOException("Expected object: " + label);
        }
        var result = new LinkedHashMap<String, Object>();
        for (var entry : map.entrySet()) {
            if (!(entry.getKey() instanceof String key)) {
                throw new IOException("Expected string key: " + label);
            }
            result.put(key, entry.getValue());
        }
        return Collections.unmodifiableMap(result);
    }

    public static List<?> array(Object value, String label) throws IOException {
        if (!(value instanceof List<?> list)) {
            throw new IOException("Expected array: " + label);
        }
        return list;
    }

    public static long integer(Object value, String label) throws IOException {
        if (!(value instanceof Long number)) {
            throw new IOException("Expected integer: " + label);
        }
        return number;
    }

    public static String string(Object value, String label) throws IOException {
        if (!(value instanceof String text)) {
            throw new IOException("Expected string: " + label);
        }
        return text;
    }
}

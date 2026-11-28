package example;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.management.ManagementFactory;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.ServiceLoader;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.dynamisengine.nn.backend.Backend;
import org.dynamisengine.nn.backend.BackendProvider;
import org.dynamisengine.nn.backend.BackendProviders;
import org.dynamisengine.nn.backend.KernelTensor;
import org.dynamisengine.nn.formats.JsonDocuments;
import org.dynamisengine.nn.generation.GenerationSession;
import org.dynamisengine.nn.generation.DeviceGenerationSession;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.pretrained.LlamaCheckpoint;
import org.dynamisengine.nn.pretrained.LlamaConfig;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;
import org.dynamisengine.nn.tokenizers.EncodeOptions;
import org.dynamisengine.nn.tokenizers.TokenizerAssets;

/** Real pinned checkpoint acceptance using a selected provider, without training or dataset modules. */
public final class PretrainedConsumer {
    private static final ValueLayout.OfFloat FLOAT = ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private record CaseResult(String prompt, int promptTokens, List<Integer> generated, String text,
                              long prefillNanos, long decodeNanos, long totalNanos) { }
    private static final class DeviceChecks {
        final AtomicBoolean active = new AtomicBoolean();
        final AtomicInteger tokenReads = new AtomicInteger(), scalarUploads = new AtomicInteger();
    }
    private static final class Errors {
        private double maximum, sumSquares;
        private long count;
        void compare(float[] actual, float[] expected, double absolute, double relative) {
            if (actual.length != expected.length) { throw new AssertionError("Logit shape differs"); }
            for (int i = 0; i < actual.length; i++) {
                double difference = Math.abs((double) actual[i] - expected[i]);
                if (!Float.isFinite(actual[i]) || !Float.isFinite(expected[i]) || difference > absolute + relative * Math.abs(expected[i])) {
                    throw new AssertionError("Logit mismatch at " + i + ": " + actual[i] + " vs " + expected[i] + ", abs=" + difference);
                }
                maximum = Math.max(maximum, difference); sumSquares += difference * difference; count++;
            }
        }
        double rms() { return Math.sqrt(sumSquares / count); }
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 1 && args[0].equals("--self-test")) {
            var config = LlamaConfig.fromJson(JsonDocuments.parse(("{\"model_type\":\"llama\",\"vocab_size\":7,\"hidden_size\":8,"
                    + "\"num_hidden_layers\":2,\"max_position_embeddings\":32,\"num_attention_heads\":2,\"num_key_value_heads\":1,\"intermediate_size\":12}").getBytes(java.nio.charset.StandardCharsets.UTF_8)));
            if (config.definition(16).blockCount() != 2) { throw new AssertionError("Pretrained configuration consumer failed"); }
            System.out.println("Pretrained JPMS consumer passed without training/data modules; real assets are opt-in"); return;
        }
        if (args.length < 1 || args.length > 3) { throw new IllegalArgumentException("Usage: PretrainedConsumer MODEL_DIRECTORY [REPORT.json [PROVIDER_ID]]"); }
        Path directory = Path.of(args[0]); Path reportPath = args.length >= 2 ? Path.of(args[1]) : directory.resolve("nnjava-validation.json");
        String provider = args.length == 3 ? args[2] : "vectrix";
        var reference = JsonDocuments.read(directory.resolve("reference.json"));
        Map<String, Object> manifest;
        try (var resource = PretrainedConsumer.class.getResourceAsStream("/smollm2-135m.json")) {
            if (resource == null) { throw new IOException("Missing packaged model manifest"); }
            manifest = JsonDocuments.parse(resource.readAllBytes());
        }
        if (!manifest.get("model").equals(reference.get("model")) || !manifest.get("revision").equals(reference.get("revision"))
                || !manifest.get("files").equals(reference.get("files"))) { throw new IOException("Reference differs from pinned checkpoint manifest"); }
        for (var entry : object(manifest, "files").entrySet()) { checkHash(directory.resolve(entry.getKey()), (String) entry.getValue()); }
        var provenance = object(reference, "reference");
        if (!"cpu".equals(provenance.get("device")) || !"float32".equals(provenance.get("dtype")) || !"eager".equals(provenance.get("attention"))) {
            throw new IOException("Expected FP32 eager CPU reference");
        }
        Path logitsPath = directory.resolve("reference-logits.f32");
        if (!"reference-logits.f32".equals(provenance.get("logits_file"))) { throw new IOException("Unexpected reference filename"); }
        checkHash(logitsPath, string(provenance, "logits_sha256"));
        int vocabulary = integer(provenance, "vocabulary_size"), rows = integer(provenance, "rows");
        long expectedBytes = Math.multiplyExact(Math.multiplyExact((long) rows, vocabulary), 4);
        if (rows < 1 || vocabulary < 1 || expectedBytes > 64 * 1024 * 1024 || Files.size(logitsPath) != expectedBytes) {
            throw new IOException("Invalid reference dimensions");
        }
        MemorySegment expectedLogits = MemorySegment.ofArray(Files.readAllBytes(logitsPath));
        var referenceErrors = new Errors(); var cacheErrors = new Errors(); var measured = new ArrayList<CaseResult>();
        var checkpoint = LlamaCheckpoint.open(directory); var config = checkpoint.configuration();
        if (config.vocabularySize() != vocabulary) { throw new AssertionError("Reference vocabulary differs"); }
        long loadStarted = System.nanoTime(); long loadNanos, parameterBytes = 0, heapAfterLoad;
        int tokenVectors, parameters; String device; var deviceChecks = new DeviceChecks();
        try (var context = open(provider, deviceChecks); var model = checkpoint.load(context, 64);
             var tokenizer = TokenizerAssets.huggingFace(directory.resolve("tokenizer.json"), TokenizerAssets.Options.gpt2())) {
            context.synchronize(); loadNanos = System.nanoTime() - loadStarted;
            device = context.device().toString();
            if (tokenizer.vocabulary().size() != vocabulary || tokenizer.vocabulary().bosTokenId().orElseThrow() != config.bosTokenId().orElseThrow()
                    || !config.eosTokenIds().contains(tokenizer.vocabulary().eosTokenId().orElseThrow())) { throw new AssertionError("Tokenizer/model control metadata differs"); }
            var allowedSpecial = EncodeOptions.plain().allowSpecial(tokenizer.vocabulary().specialTokens().keySet().toArray(String[]::new));
            var vectors = JsonDocuments.array(reference.get("tokenization"), "tokenization"); tokenVectors = vectors.size();
            for (Object value : vectors) {
                var vector = JsonDocuments.object(value, "tokenization vector"); int[] ids = integers(vector.get("ids"));
                equal(ids, tokenizer.encode(string(vector, "text"), allowedSpecial), "Reference tokenization");
                if (!string(vector, "decoded").equals(tokenizer.decode(ids))) { throw new AssertionError("Reference decoded text differs"); }
            }
            Set<Long> unique = new HashSet<>();
            for (var parameter : model.namedParameters().values()) {
                try (var tensor = parameter.value()) { if (unique.add(tensor.id())) { parameterBytes += tensor.shape().elements() * tensor.dtype().bytes(); } }
            }
            parameters = unique.size(); heapAfterLoad = ManagementFactory.getMemoryMXBean().getHeapMemoryUsage().getUsed();
            model.training(false); int baseline = context.liveAllocations();
            double absolute = number(object(reference, "tolerance"), "absolute"), relative = number(object(reference, "tolerance"), "relative");
            if (absolute <= 0 || absolute > .003 || relative < 0 || relative > .0003) { throw new IOException("Reference tolerances exceed acceptance contract"); }
            for (Object value : JsonDocuments.array(reference.get("cases"), "cases")) {
                var test = JsonDocuments.object(value, "case"); int[] prompt = integers(test.get("ids")), generated = integers(test.get("generated_ids"));
                var rowMetadata = JsonDocuments.array(test.get("rows"), "rows");
                if (generated.length != 4 || rowMetadata.size() != 5 || prompt.length == 0) { throw new IOException("Expected a four-token reference case"); }
                float[] last;
                try (var session = model.newSession(1)) {
                    try (var input = context.tensor(prompt, Shape.of(1, prompt.length)); var logits = session.prefill(input)) { last = last(logits); }
                    float[] firstPrefill = last;
                    for (int step = 0; step <= generated.length; step++) {
                        var row = JsonDocuments.object(rowMetadata.get(step), "row");
                        float[] expected = row(expectedLogits, integer(row, "row"), rows, vocabulary);
                        referenceErrors.compare(last, expected, absolute, relative);
                        if (argmax(last) != integer(row, "argmax")) { throw new AssertionError("Greedy argmax differs from reference"); }
                        if (step == generated.length) { break; }
                        if (argmax(last) != generated[step]) { throw new AssertionError("Generated reference ID differs"); }
                        try (var input = context.tensor(new int[]{generated[step]}, Shape.of(1, 1)); var logits = session.decode(input)) { last = last(logits); }
                    }
                    if (session.length() != prompt.length + generated.length) { throw new AssertionError("Cache did not grow with accepted tokens"); }
                    int[] all = Arrays.copyOf(prompt, prompt.length + generated.length); System.arraycopy(generated, 0, all, prompt.length, generated.length);
                    try (var input = context.tensor(all, Shape.of(1, all.length)); var logits = model.forward(input)) {
                        cacheErrors.compare(last, last(logits), absolute, relative);
                    }
                    session.reset();
                    try (var input = context.tensor(prompt, Shape.of(1, prompt.length)); var logits = session.prefill(input)) {
                        cacheErrors.compare(last(logits), firstPrefill, absolute, relative);
                    }
                }
                try (var generation = GenerationSession.builder().model(model).maxNewTokens(4).stopTokenIds(config.eosTokenIds().stream().mapToInt(Integer::intValue).toArray())
                        .sampling(s -> s.temperature(0)).build(); var decoder = tokenizer.newDecoder()) {
                    var streamed = new StringBuilder();
                    var result = generation.generate(prompt, token -> { streamed.append(decoder.accept(token.tokenId())); return GenerationSession.Decision.CONTINUE; });
                    streamed.append(decoder.finish()); equal(generated, result.tokenIds(), "Public generation");
                    String text = tokenizer.decode(result.tokenIds());
                    if (!text.equals(string(test, "generated_text")) || !text.contentEquals(streamed)) { throw new AssertionError("Streamed/whole generated text differs"); }
                    measured.add(new CaseResult(string(test, "text"), prompt.length, Arrays.stream(result.tokenIds()).boxed().toList(), text,
                            result.prefillNanos(), result.decodeNanos(), result.totalNanos()));
                    System.out.println("Verified: " + test.get("text") + " -> " + text);
                }
                if (!provider.equals("vectrix")) {
                    deviceChecks.active.set(true);
                    try (var generation = DeviceGenerationSession.builder().model(model).maxNewTokens(4).sampling(s -> s.temperature(0))
                            .stopTokenIds(config.eosTokenIds().stream().mapToInt(Integer::intValue).toArray()).build(); var decoder = tokenizer.newDecoder()) {
                        var streamed = new StringBuilder();
                        var result = generation.generate(prompt, token -> { streamed.append(decoder.accept(token.tokenId())); return GenerationSession.Decision.CONTINUE; });
                        streamed.append(decoder.finish()); equal(generated, result.tokenIds(), "Device-sampled generation");
                        if (!string(test, "generated_text").contentEquals(streamed)) { throw new AssertionError("Device-sampled streaming text differs"); }
                    } finally { deviceChecks.active.set(false); }
                    System.out.println("Device sampling verified without logit readback: " + test.get("text"));
                }
                if (context.liveAllocations() != baseline) { throw new AssertionError("Inference leaked allocations"); }
            }
        }
        if (!provider.equals("vectrix") && (deviceChecks.tokenReads.get() != 12 || deviceChecks.scalarUploads.get() != 0)) { throw new AssertionError("Device generation transfer contract differs"); }
        writeReport(reportPath, manifest, referenceErrors, cacheErrors, tokenVectors, parameters, parameterBytes, heapAfterLoad, loadNanos, measured, provider, device, deviceChecks);
        System.out.printf("Validated %,d logits; max absolute error %.8g; report: %s%n", referenceErrors.count, referenceErrors.maximum, reportPath);
    }

    private static void writeReport(Path path, Map<String, Object> manifest, Errors reference, Errors cache,
                                    int vectors, int parameters, long parameterBytes, long heap, long loadNanos, List<CaseResult> cases,
                                    String provider, String device, DeviceChecks deviceChecks) throws IOException {
        try (var generator = new JsonFactory().createGenerator(Files.newOutputStream(path))) {
            generator.useDefaultPrettyPrinter(); generator.writeStartObject();
            generator.writeStringField("model", (String) manifest.get("model")); generator.writeStringField("revision", (String) manifest.get("revision"));
            generator.writeStringField("backend", device); generator.writeStringField("provider", provider);
            generator.writeStringField("vectrix", "4.1.0"); generator.writeStringField("java", System.getProperty("java.version"));
            if (!provider.equals("vectrix")) { generator.writeStringField("dynamisgpu", "1.14.1"); }
            generator.writeBooleanField("simd", ModuleLayer.boot().findModule("jdk.incubator.vector").isPresent());
            generator.writeStringField("os", System.getProperty("os.name") + " " + System.getProperty("os.arch"));
            generator.writeNumberField("tokenization_vectors", vectors); generator.writeNumberField("unique_parameters", parameters);
            generator.writeNumberField("parameter_bytes", parameterBytes); generator.writeNumberField("heap_used_after_load", heap);
            generator.writeNumberField("context_capacity", 64); generator.writeNumberField("load_ms", loadNanos / 1e6);
            generator.writeNumberField("compared_logits", reference.count); generator.writeNumberField("max_absolute_logit_error", reference.maximum);
            generator.writeNumberField("rms_logit_error", reference.rms()); generator.writeNumberField("cache_check_max_absolute_error", cache.maximum);
            generator.writeBooleanField("device_sampling_verified", !provider.equals("vectrix"));
            generator.writeNumberField("device_generation_token_reads", deviceChecks.tokenReads.get());
            generator.writeNumberField("device_generation_scalar_token_uploads", deviceChecks.scalarUploads.get());
            generator.writeStringField("timing_policy", "One request per prompt after correctness passes; includes accepted final-token decode and host sampling/logit readback; not a throughput benchmark");
            generator.writeArrayFieldStart("cases");
            for (var result : cases) {
                generator.writeStartObject(); generator.writeStringField("prompt", result.prompt); generator.writeNumberField("prompt_tokens", result.promptTokens);
                generator.writeArrayFieldStart("generated_ids"); for (int id : result.generated) { generator.writeNumber(id); } generator.writeEndArray();
                generator.writeStringField("generated_text", result.text); generator.writeNumberField("prefill_ms", result.prefillNanos / 1e6);
                generator.writeNumberField("decode_ms", result.decodeNanos / 1e6); generator.writeNumberField("total_ms", result.totalNanos / 1e6);
                generator.writeEndObject();
            }
            generator.writeEndArray(); generator.writeEndObject();
        }
    }
    private static ExecutionContext open(String id, DeviceChecks checks) {
        if (id.equals("vectrix")) { return ExecutionContext.cpu(); }
        var provider = ServiceLoader.load(BackendProvider.class).stream().map(ServiceLoader.Provider::get)
                .filter(value -> value.id().equals(id)).findFirst().orElseThrow(() -> new IllegalArgumentException("Provider unavailable: " + id));
        var devices = provider.devices();
        if (devices.isEmpty()) { throw new IllegalStateException("Requested provider has no device: " + id); }
        Backend backend = BackendProviders.open(id, devices.getFirst().id());
        var monitored = (Backend) Proxy.newProxyInstance(Backend.class.getClassLoader(), new Class<?>[]{Backend.class}, (proxy, method, args) -> {
            if (checks.active.get() && method.getName().equals("download")) {
                if (args[1] instanceof float[]) { throw new AssertionError("Device generation downloaded floating-point logits"); }
                if (args[1] instanceof int[] values) {
                    if (values.length != 1 || ((KernelTensor) args[0]).buffer().elements() != 1) { throw new AssertionError("Expected a scalar token readback"); }
                    checks.tokenReads.incrementAndGet();
                }
            }
            if (checks.active.get() && method.getName().equals("upload") && args[1] instanceof int[] values && values.length == 1) { checks.scalarUploads.incrementAndGet(); }
            try { return method.invoke(backend, args); } catch (InvocationTargetException error) { throw error.getCause(); }
        });
        return ExecutionContext.create(monitored);
    }
    private static float[] row(MemorySegment data, int row, int rows, int vocabulary) throws IOException {
        if (row < 0 || row >= rows) { throw new IOException("Reference row outside data"); }
        float[] values = new float[vocabulary];
        for (int i = 0; i < vocabulary; i++) { values[i] = data.get(FLOAT, ((long) row * vocabulary + i) * 4); } return values;
    }
    private static float[] last(Tensor logits) { long time = logits.shape().dimension(1); try (var row = Ops.slice(logits, 1, time - 1, time)) { return row.toFloatArray(); } }
    private static int argmax(float[] values) { int best = 0; for (int i = 1; i < values.length; i++) { if (values[i] > values[best]) { best = i; } } return best; }
    private static void equal(int[] expected, int[] actual, String label) { if (!Arrays.equals(expected, actual)) { throw new AssertionError(label + " differs: " + Arrays.toString(actual)); } }
    private static Map<String, Object> object(Map<String, Object> json, String key) throws IOException { return JsonDocuments.object(json.get(key), key); }
    private static String string(Map<String, Object> json, String key) throws IOException { return JsonDocuments.string(json.get(key), key); }
    private static int integer(Map<String, Object> json, String key) throws IOException { return Math.toIntExact(JsonDocuments.integer(json.get(key), key)); }
    private static double number(Map<String, Object> json, String key) throws IOException { if (!(json.get(key) instanceof Number n)) { throw new IOException("Expected number: " + key); } return n.doubleValue(); }
    private static int[] integers(Object value) throws IOException {
        var list = JsonDocuments.array(value, "token IDs"); int[] result = new int[list.size()];
        for (int i = 0; i < result.length; i++) { result[i] = Math.toIntExact(JsonDocuments.integer(list.get(i), "token ID")); } return result;
    }
    private static void checkHash(Path path, String expected) throws Exception {
        var digest = MessageDigest.getInstance("SHA-256");
        try (var input = Files.newInputStream(path)) { byte[] buffer = new byte[1024 * 1024]; int count; while ((count = input.read(buffer)) != -1) { digest.update(buffer, 0, count); } }
        if (!HexFormat.of().formatHex(digest.digest()).equals(expected)) { throw new IOException("Checksum mismatch: " + path); }
    }
}

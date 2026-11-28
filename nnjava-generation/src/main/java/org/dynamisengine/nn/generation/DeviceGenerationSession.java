package org.dynamisengine.nn.generation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;
import org.dynamisengine.nn.backend.Operation;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.ops.SamplingOptions;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;
import org.dynamisengine.nn.transformer.CausalLanguageModel;

/** Explicit provider sampling: tensor logits/token feedback, with only token IDs read for host control. */
public final class DeviceGenerationSession implements AutoCloseable {
    public record GeneratedToken(int tokenId, int index) { }
    @FunctionalInterface public interface TokenListener { GenerationSession.Decision onToken(GeneratedToken token); }
    public record Result(List<Integer> prompt, List<GeneratedToken> tokens, GenerationSession.StopReason reason, long totalNanos) {
        public Result { prompt = List.copyOf(prompt); tokens = List.copyOf(tokens); Objects.requireNonNull(reason); }
        public int[] tokenIds() { return tokens.stream().mapToInt(GeneratedToken::tokenId).toArray(); }
        public int[] allTokenIds() { return java.util.stream.IntStream.concat(prompt.stream().mapToInt(Integer::intValue), Arrays.stream(tokenIds())).toArray(); }
    }

    private final Thread owner = Thread.currentThread();
    private final TensorAutoregressiveModel model;
    private final ExecutionContext context;
    private final GenerationConfig configuration;
    private final SamplingOptions sampling;
    private final int vocabulary, capacity;
    private final List<StoppingCriterion> criteria;
    private final AtomicBoolean cancelled = new AtomicBoolean();
    private boolean closed, running;

    private DeviceGenerationSession(Builder builder) {
        model = Objects.requireNonNull(builder.model, "model"); context = Objects.requireNonNull(model.executionContext());
        context.ensureOpen(); configuration = builder.define(); criteria = List.copyOf(builder.criteria);
        vocabulary = model.vocabularySize(); capacity = model.contextLength();
        if (vocabulary < 1 || capacity < 1) { throw new IllegalArgumentException("Positive model vocabulary and context capacity required"); }
        var parameters = configuration.sampling();
        if (parameters.repetitionPenalty() != 1 || parameters.presencePenalty() != 0 || parameters.frequencyPenalty() != 0) {
            throw new UnsupportedOperationException("Device generation does not implement repetition/presence/frequency penalties");
        }
        sampling = new SamplingOptions(parameters.temperature(), parameters.topK(), parameters.topP(), parameters.seed());
        if (!context.supportsSampling(sampling, 1, vocabulary)) { throw new UnsupportedOperationException("Provider cannot execute the requested sampling policy"); }
        configuration.stopTokenIds().forEach(this::checkToken); configuration.stopSequences().forEach(ids -> ids.forEach(this::checkToken));
        if (configuration.minNewTokens() > 0 && !configuration.stopTokenIds().isEmpty()
                && (!context.supports(Operation.WHERE, DType.FP32) || !context.supports(Operation.FILL, DType.FP32))) {
            throw new UnsupportedOperationException("Minimum-length stop suppression requires provider selection/fill");
        }
    }

    public static Builder builder() { return new Builder(); }
    public GenerationConfig configuration() { ensureOpen(); return configuration; }
    /** Cross-thread cancellation is observed between complete provider calls. */
    public void cancel() { cancelled.set(true); }
    public Result generate(int[] prompt) { return generate(prompt, token -> GenerationSession.Decision.CONTINUE); }
    public Result generate(int[] prompt, TokenListener listener) {
        ensureOpen(); Objects.requireNonNull(listener);
        if (running) { throw new IllegalStateException("Generation callbacks cannot reenter the session"); }
        if (context.activeTape() != null) { throw new IllegalStateException("Device generation is inference-only"); }
        int[] input = Objects.requireNonNull(prompt).clone();
        if (input.length == 0 || (long) input.length + configuration.maxNewTokens() > capacity) {
            throw new IllegalArgumentException("Prompt plus generation budget must fit context capacity=" + capacity);
        }
        for (int token : input) { checkToken(token); }
        running = true; long started = System.nanoTime();
        try {
            var result = execute(input, listener, started);
            return new Result(result.prompt(), result.tokens(), result.reason(), System.nanoTime() - started);
        }
        finally { running = false; cancelled.set(false); }
    }

    private Result execute(int[] prompt, TokenListener listener, long started) {
        var generated = new ArrayList<GeneratedToken>();
        if (cancelled.get()) { return result(prompt, generated, GenerationSession.StopReason.CANCELLED, started); }
        if (configuration.maxNewTokens() == 0) { return result(prompt, generated, GenerationSession.StopReason.MAX_NEW_TOKENS, started); }
        int[] history = Arrays.copyOf(prompt, prompt.length + configuration.maxNewTokens());
        boolean suppress = configuration.minNewTokens() > 0 && !configuration.stopTokenIds().isEmpty();
        boolean[] allowed = suppress ? new boolean[vocabulary] : null;
        if (suppress) { Arrays.fill(allowed, true); configuration.stopTokenIds().forEach(id -> allowed[id] = false); }
        try (var cursor = Objects.requireNonNull(model.openSession(), "model cursor");
             var input = context.tensor(prompt, Shape.of(1, prompt.length));
             var mask = suppress ? context.tensor(allowed, Shape.of(vocabulary)) : null;
             var excluded = suppress ? context.full(Shape.of(), Float.NEGATIVE_INFINITY) : null) {
            if (cursor.length() != 0) { throw new IllegalStateException("A new model cursor must be empty"); }
            Tensor logits = Objects.requireNonNull(cursor.prefill(input), "prefill logits");
            try {
                validate(logits);
                context.synchronize();
                if (cursor.length() != prompt.length) { throw new IllegalStateException("Cursor did not commit the prompt"); }
                for (int step = 0; step < configuration.maxNewTokens(); step++) {
                    if (cancelled.get()) { return result(prompt, generated, GenerationSession.StopReason.CANCELLED, started); }
                    try (var working = suppress && step < configuration.minNewTokens() ? Ops.where(mask, logits, excluded) : logits.retain();
                         var selected = Ops.sample(working, sampling, step); var tokenInput = Ops.reshape(selected, Shape.of(1, 1))) {
                        int id = selected.toIntArray()[0]; checkToken(id); // Four bytes for callbacks/stopping; never a vocabulary logit row.
                        if (cancelled.get()) { return result(prompt, generated, GenerationSession.StopReason.CANCELLED, started); }
                        Tensor next = Objects.requireNonNull(cursor.decode(tokenInput), "decode logits");
                        try { validate(next); context.synchronize(); } catch (RuntimeException | Error error) { next.close(); throw error; }
                        logits.close(); logits = next;
                        if (cursor.length() != prompt.length + step + 1) { throw new IllegalStateException("Cursor did not commit the generated token"); }
                        history[prompt.length + step] = id; var token = new GeneratedToken(id, step); generated.add(token);
                        var reason = stopReason(history, prompt.length, step + 1);
                        var decision = Objects.requireNonNull(listener.onToken(token), "listener decision");
                        if (reason == null && decision == GenerationSession.Decision.STOP) { reason = GenerationSession.StopReason.CALLBACK; }
                        if (reason == null && cancelled.get()) { reason = GenerationSession.StopReason.CANCELLED; }
                        if (reason == null && !criteria.isEmpty()) {
                            var after = new TokenHistory(Arrays.copyOf(history, prompt.length + step + 1), prompt.length);
                            for (var criterion : criteria) { if (criterion.shouldStop(after)) { reason = GenerationSession.StopReason.CRITERION; break; } }
                        }
                        if (reason != null) { return result(prompt, generated, reason, started); }
                    }
                }
                return result(prompt, generated, GenerationSession.StopReason.MAX_NEW_TOKENS, started);
            } finally { logits.close(); }
        }
    }

    private GenerationSession.StopReason stopReason(int[] history, int prompt, int count) {
        if (configuration.stopTokenIds().contains(history[prompt + count - 1])) { return GenerationSession.StopReason.STOP_TOKEN; }
        if (count < configuration.minNewTokens()) { return null; }
        for (var sequence : configuration.stopSequences()) {
            if (sequence.size() > count) { continue; }
            boolean matches = true;
            for (int i = 0; i < sequence.size(); i++) { matches &= history[prompt + count - sequence.size() + i] == sequence.get(i); }
            if (matches) { return GenerationSession.StopReason.STOP_SEQUENCE; }
        }
        return null;
    }
    private void validate(Tensor logits) {
        context.requireContext(logits);
        if (logits.dtype() != DType.FP32 || !logits.shape().equals(Shape.of(vocabulary))) { throw new IllegalArgumentException("Expected FP32 [vocabulary] logits"); }
    }
    private void checkToken(int id) { if (id < 0 || id >= vocabulary) { throw new IllegalArgumentException("Token outside model vocabulary"); } }
    private Result result(int[] prompt, List<GeneratedToken> tokens, GenerationSession.StopReason reason, long started) {
        return new Result(Arrays.stream(prompt).boxed().toList(), tokens, reason, System.nanoTime() - started);
    }
    private void ensureOpen() {
        if (Thread.currentThread() != owner) { throw new IllegalStateException("Generation session is thread-confined"); }
        if (closed) { throw new IllegalStateException("Generation session is closed"); }
        context.ensureOpen();
    }
    @Override public void close() {
        if (Thread.currentThread() != owner) { throw new IllegalStateException("Generation session is thread-confined"); }
        if (running) { throw new IllegalStateException("Cancel a running session before closing it"); } closed = true;
    }

    public static final class Builder {
        private TensorAutoregressiveModel model;
        private SamplingParameters sampling = SamplingParameters.builder().define();
        private int maximum = 128, minimum;
        private final Set<Integer> stops = new HashSet<>();
        private final List<List<Integer>> sequences = new ArrayList<>();
        private final List<StoppingCriterion> criteria = new ArrayList<>();
        private Builder() { }
        public Builder model(TensorAutoregressiveModel value) { model = Objects.requireNonNull(value); return this; }
        public Builder model(CausalLanguageModel value) { return model(CausalModels.adaptDevice(value)); }
        public Builder model(CausalLanguageModel value, org.dynamisengine.nn.transformer.InferenceOptions options) {
            return model(CausalModels.adaptDevice(value, options));
        }
        public Builder sampling(SamplingParameters value) { sampling = Objects.requireNonNull(value); return this; }
        public Builder sampling(Consumer<SamplingParameters.Builder> settings) {
            var builder = sampling.toBuilder(); settings.accept(builder); sampling = builder.define(); return this;
        }
        public Builder maxNewTokens(int value) { maximum = value; return this; }
        public Builder minNewTokens(int value) { minimum = value; return this; }
        public Builder stopTokenIds(int... ids) { for (int id : ids) { stops.add(id); } return this; }
        public Builder stopSequence(int... ids) { sequences.add(Arrays.stream(ids).boxed().toList()); return this; }
        public Builder stopping(StoppingCriterion criterion) { criteria.add(Objects.requireNonNull(criterion)); return this; }
        public GenerationConfig define() { return new GenerationConfig(sampling, maximum, minimum, stops, sequences); }
        public Builder configuration(GenerationConfig value) {
            sampling = value.sampling(); maximum = value.maxNewTokens(); minimum = value.minNewTokens(); stops.clear(); stops.addAll(value.stopTokenIds());
            sequences.clear(); sequences.addAll(value.stopSequences()); return this;
        }
        public DeviceGenerationSession build() { return new DeviceGenerationSession(this); }
    }
}

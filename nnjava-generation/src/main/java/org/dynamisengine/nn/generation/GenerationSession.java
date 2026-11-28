package org.dynamisengine.nn.generation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;
import org.dynamisengine.nn.transformer.CausalLanguageModel;

/**
 * Single-request token generation. Borrows its model; each request owns and closes
 * a fresh cursor. Host sampling is explicit; no tokenizer or training dependency.
 */
public final class GenerationSession implements AutoCloseable {
    public enum Decision { CONTINUE, STOP }
    public enum StopReason { MAX_NEW_TOKENS, STOP_TOKEN, STOP_SEQUENCE, CRITERION, CALLBACK, CANCELLED }
    public record GeneratedToken(int tokenId,int index,double logProbability) { }
    @FunctionalInterface public interface TokenListener { Decision onToken(GeneratedToken token); }

    public record Result(List<Integer> prompt,List<GeneratedToken> tokens,StopReason reason,long prefillNanos,long decodeNanos,long totalNanos) {
        public Result { prompt=List.copyOf(prompt); tokens=List.copyOf(tokens); Objects.requireNonNull(reason); }
        public int[] tokenIds() { return tokens.stream().mapToInt(GeneratedToken::tokenId).toArray(); }
        public int[] allTokenIds() { return java.util.stream.IntStream.concat(prompt.stream().mapToInt(Integer::intValue),Arrays.stream(tokenIds())).toArray(); }
    }

    private final Thread owner=Thread.currentThread();
    private final AutoregressiveModel model;
    private final GenerationConfig configuration;
    private final int vocabularySize,contextLength;
    private final HostSampler sampler;
    private final List<LogitsProcessor> processors;
    private final List<StoppingCriterion> criteria;
    private final AtomicBoolean cancelled=new AtomicBoolean();
    private boolean closed,running;

    private GenerationSession(Builder builder) {
        model=Objects.requireNonNull(builder.model,"model"); configuration=builder.define();
        processors=List.copyOf(builder.processors); criteria=List.copyOf(builder.criteria);
        vocabularySize=model.vocabularySize(); contextLength=model.contextLength();
        if (vocabularySize<1 || contextLength<1) { throw new IllegalArgumentException("Model vocabulary/context metadata must be positive"); }
        configuration.stopTokenIds().forEach(this::checkToken);
        configuration.stopSequences().forEach(ids -> ids.forEach(this::checkToken));
        sampler=new HostSampler(vocabularySize,configuration.sampling());
    }

    public static Builder builder() { return new Builder(); }
    public GenerationConfig configuration() { ensureOpen(); return configuration; }
    /** Cross-thread cancellation is observed between complete prefill/decode calls. */
    public void cancel() { cancelled.set(true); }

    public Result generate(int[] prompt) { return generate(prompt,token -> Decision.CONTINUE); }

    /** Each call restarts the configured Philox stream and creates an independent cache. */
    public Result generate(int[] prompt,TokenListener listener) {
        ensureOpen();
        if (running) { throw new IllegalStateException("Generation callbacks cannot reenter the session"); }
        Objects.requireNonNull(listener); int[] input=Objects.requireNonNull(prompt).clone();
        if (input.length==0 || (long)input.length+configuration.maxNewTokens()>contextLength) {
            throw new IllegalArgumentException("Nonempty prompt plus generation budget must fit contextLength="+contextLength);
        }
        for (int token:input) { checkToken(token); }
        running=true;
        long started=System.nanoTime();
        try {
            var completed=execute(input,listener,started);
            return new Result(completed.prompt(),completed.tokens(),completed.reason(),completed.prefillNanos(),completed.decodeNanos(),System.nanoTime()-started);
        } finally { running=false; cancelled.set(false); }
    }

    private Result execute(int[] input,TokenListener listener,long started) {
        long prefillNanos=0,decodeNanos=0;
        var generated=new ArrayList<GeneratedToken>();
            if (cancelled.get()) { return result(input,generated,StopReason.CANCELLED,0,0,started); }
            if (configuration.maxNewTokens()==0) { return result(input,generated,StopReason.MAX_NEW_TOKENS,0,0,started); }
            int[] history=Arrays.copyOf(input,input.length+configuration.maxNewTokens()),counts=new int[vocabularySize];
            for (int id:input) { counts[id]++; }
            long prefillStarted=System.nanoTime();
            try (var cursor=Objects.requireNonNull(model.openSession(),"model cursor")) {
                if (cursor.length()!=0) { throw new IllegalStateException("A new model cursor must be empty"); }
                float[] logits=cursor.prefill(input.clone()); validateLogits(logits);
                if (cursor.length()!=input.length) { throw new IllegalStateException("Cursor did not commit the prompt"); }
                prefillNanos=System.nanoTime()-prefillStarted;
                for (int step=0;step<configuration.maxNewTokens();step++) {
                    if (cancelled.get()) { return result(input,generated,StopReason.CANCELLED,prefillNanos,decodeNanos,started); }
                    float[] working=logits.clone();
                    penalties(working,counts);
                    var before=new TokenHistory(Arrays.copyOf(history,input.length+step),input.length);
                    for (var processor:processors) { processor.process(before,working); }
                    if (step<configuration.minNewTokens()) { for (int id:configuration.stopTokenIds()) { working[id]=Float.NEGATIVE_INFINITY; } }
                    var choice=sampler.sample(working,step);
                    if (cancelled.get()) { return result(input,generated,StopReason.CANCELLED,prefillNanos,decodeNanos,started); }
                    long decodeStarted=System.nanoTime();
                    logits=cursor.decode(choice.tokenId()); validateLogits(logits);
                    decodeNanos+=System.nanoTime()-decodeStarted;
                    if (cursor.length()!=input.length+step+1) { throw new IllegalStateException("Cursor did not commit the generated token"); }
                    history[input.length+step]=choice.tokenId(); counts[choice.tokenId()]++;
                    var token=new GeneratedToken(choice.tokenId(),step,choice.logProbability()); generated.add(token);
                    var reason=stopReason(history,input.length,step+1);
                    var decision=Objects.requireNonNull(listener.onToken(token),"token listener decision");
                    if (reason==null && decision==Decision.STOP) { reason=StopReason.CALLBACK; }
                    if (reason==null && cancelled.get()) { reason=StopReason.CANCELLED; }
                    if (reason==null && !criteria.isEmpty()) {
                        var after=new TokenHistory(Arrays.copyOf(history,input.length+step+1),input.length);
                        for (var criterion:criteria) { if (criterion.shouldStop(after)) { reason=StopReason.CRITERION; break; } }
                    }
                    if (reason!=null) { return result(input,generated,reason,prefillNanos,decodeNanos,started); }
                }
                return result(input,generated,StopReason.MAX_NEW_TOKENS,prefillNanos,decodeNanos,started);
            }
    }

    private void penalties(float[] logits,int[] counts) {
        var p=configuration.sampling();
        for (int id=0;id<counts.length;id++) {
            if (counts[id]==0 || logits[id]==Float.NEGATIVE_INFINITY) { continue; }
            double value=logits[id]>=0 ? logits[id]/(double)p.repetitionPenalty() : logits[id]*(double)p.repetitionPenalty();
            value-=p.presencePenalty()+counts[id]*(double)p.frequencyPenalty();
            if (!Double.isFinite(value) || Math.abs(value)>Float.MAX_VALUE) { throw new ArithmeticException("Logits penalty overflow for token "+id); }
            logits[id]=(float)value;
        }
    }

    private StopReason stopReason(int[] history,int promptLength,int generated) {
        if (configuration.stopTokenIds().contains(history[promptLength+generated-1])) { return StopReason.STOP_TOKEN; }
        if (generated<configuration.minNewTokens()) { return null; }
        for (var sequence:configuration.stopSequences()) {
            if (sequence.size()>generated) { continue; }
            int start=promptLength+generated-sequence.size(); boolean matches=true;
            for (int i=0;i<sequence.size();i++) { matches &= history[start+i]==sequence.get(i); }
            if (matches) { return StopReason.STOP_SEQUENCE; }
        }
        return null;
    }

    private Result result(int[] prompt,List<GeneratedToken> tokens,StopReason reason,long prefill,long decode,long started) {
        return new Result(Arrays.stream(prompt).boxed().toList(),tokens,reason,prefill,decode,System.nanoTime()-started);
    }
    private void validateLogits(float[] logits) {
        if (logits==null || logits.length!=vocabularySize) { throw new IllegalArgumentException("Model returned the wrong logits width"); }
        for (float value:logits) { if (Float.isNaN(value) || value==Float.POSITIVE_INFINITY) { throw new IllegalArgumentException("Model returned nonfinite logits"); } }
    }
    private void checkToken(int id) { if (id<0 || id>=vocabularySize) { throw new IllegalArgumentException("Token ID outside model vocabulary: "+id); } }
    private void ensureOpen() {
        if (Thread.currentThread()!=owner) { throw new IllegalStateException("Generation session is thread-confined"); }
        if (closed) { throw new IllegalStateException("Generation session is closed"); }
    }
    @Override public void close() {
        if (Thread.currentThread()!=owner) { throw new IllegalStateException("Generation session is thread-confined"); }
        if (!closed) { if (running) { throw new IllegalStateException("Cannot close a running session; cancel instead"); } closed=true; }
    }

    public static final class Builder {
        private AutoregressiveModel model;
        private SamplingParameters sampling=SamplingParameters.builder().define();
        private int maximum=128,minimum;
        private final Set<Integer> stops=new HashSet<>();
        private final List<List<Integer>> sequences=new ArrayList<>();
        private final List<LogitsProcessor> processors=new ArrayList<>();
        private final List<StoppingCriterion> criteria=new ArrayList<>();
        private Builder() { }
        public Builder model(AutoregressiveModel value) { model=Objects.requireNonNull(value); return this; }
        public Builder model(CausalLanguageModel value) { return model(CausalModels.adapt(value)); }
        public Builder sampling(SamplingParameters value) { sampling=Objects.requireNonNull(value); return this; }
        public Builder sampling(Consumer<SamplingParameters.Builder> settings) {
            var builder=sampling.toBuilder(); settings.accept(builder); sampling=builder.define(); return this;
        }
        public Builder maxNewTokens(int value) { maximum=value; return this; }
        public Builder minNewTokens(int value) { minimum=value; return this; }
        public Builder stopTokenIds(int... values) { for (int id:values) { stops.add(id); } return this; }
        public Builder stopSequence(int... values) { sequences.add(Arrays.stream(values).boxed().toList()); return this; }
        public Builder processor(LogitsProcessor value) { processors.add(Objects.requireNonNull(value)); return this; }
        public Builder stopping(StoppingCriterion value) { criteria.add(Objects.requireNonNull(value)); return this; }
        /** Standard settings only; custom hooks and the model remain application-owned. */
        public GenerationConfig define() { return new GenerationConfig(sampling,maximum,minimum,stops,sequences); }
        public Builder configuration(GenerationConfig value) {
            sampling=value.sampling(); maximum=value.maxNewTokens(); minimum=value.minNewTokens();
            stops.clear(); stops.addAll(value.stopTokenIds()); sequences.clear(); sequences.addAll(value.stopSequences()); return this;
        }
        public GenerationSession build() { return new GenerationSession(this); }
    }
}

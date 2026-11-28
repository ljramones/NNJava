package org.dynamisengine.nn.generation;

/** Immutable host-sampling policy. Top-p is applied to the renormalized top-k set. */
public record SamplingParameters(float temperature,int topK,float topP,float repetitionPenalty,
                                  float presencePenalty,float frequencyPenalty,long seed) {
    public SamplingParameters {
        if (!Float.isFinite(temperature) || temperature<0 || topK<0 || !Float.isFinite(topP) || topP<=0 || topP>1
                || !Float.isFinite(repetitionPenalty) || repetitionPenalty<=0 || !Float.isFinite(presencePenalty) || !Float.isFinite(frequencyPenalty)) {
            throw new IllegalArgumentException("Invalid temperature, top-k/top-p, or repetition penalties");
        }
    }
    public static Builder builder() { return new Builder(); }
    public Builder toBuilder() {
        return builder().temperature(temperature).topK(topK).topP(topP).repetitionPenalty(repetitionPenalty)
                .presencePenalty(presencePenalty).frequencyPenalty(frequencyPenalty).seed(seed);
    }
    public static final class Builder {
        private float temperature=1,topP=1,repetitionPenalty=1,presencePenalty,frequencyPenalty;
        private int topK;
        private long seed;
        public Builder temperature(float value) { temperature=value; return this; }
        public Builder topK(int value) { topK=value; return this; }
        public Builder topP(float value) { topP=value; return this; }
        public Builder repetitionPenalty(float value) { repetitionPenalty=value; return this; }
        public Builder presencePenalty(float value) { presencePenalty=value; return this; }
        public Builder frequencyPenalty(float value) { frequencyPenalty=value; return this; }
        public Builder seed(long value) { seed=value; return this; }
        public SamplingParameters define() { return new SamplingParameters(temperature,topK,topP,repetitionPenalty,presencePenalty,frequencyPenalty,seed); }
    }
}

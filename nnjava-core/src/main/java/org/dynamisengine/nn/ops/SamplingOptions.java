package org.dynamisengine.nn.ops;

/** Provider sampling options. The provider publishes the draw/order contract; seed streams are not interchangeable. */
public record SamplingOptions(float temperature, int topK, float topP, long seed) {
    public SamplingOptions {
        if (!Float.isFinite(temperature) || temperature < 0 || topK < 0 || !Float.isFinite(topP) || topP <= 0 || topP > 1) {
            throw new IllegalArgumentException("Invalid sampling temperature/top-k/top-p");
        }
    }
    public static SamplingOptions greedy() { return new SamplingOptions(0, 0, 1, 0); }
    public boolean greedySelection() { return temperature == 0 || topK == 1; }
}

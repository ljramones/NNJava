package org.dynamisengine.nn.generation;

import java.util.Objects;
import org.dynamisengine.vectrix.hash.Philox;
import org.dynamisengine.vectrix.nn.Sampler;

/** Thread-confined host sampler, explicitly using Vectrix probability/ranking and Philox kernels. */
public final class HostSampler {
    public record Choice(int tokenId,double logProbability) { }
    private final Thread owner=Thread.currentThread();
    private final SamplingParameters parameters;
    private final float[] working,probabilities;
    private final int[] order;

    public HostSampler(int vocabularySize,SamplingParameters parameters) {
        if (vocabularySize<1) { throw new IllegalArgumentException("Positive vocabulary size required"); }
        this.parameters=Objects.requireNonNull(parameters);
        working=new float[vocabularySize]; probabilities=new float[vocabularySize]; order=new int[vocabularySize];
    }

    /** Log probability belongs to the processed, temperature-scaled, filtered distribution. Greedy has logP 0. */
    public Choice sample(float[] logits,long sampleIndex) {
        if (Thread.currentThread()!=owner) { throw new IllegalStateException("Sampler is thread-confined"); }
        if (logits.length!=working.length || sampleIndex<0) { throw new IllegalArgumentException("Invalid logits width or sampling index"); }
        float maximum=Float.NEGATIVE_INFINITY;
        for (float value:logits) {
            if (Float.isNaN(value) || value==Float.POSITIVE_INFINITY) { throw new IllegalArgumentException("Logits cannot contain NaN or positive infinity"); }
            maximum=Math.max(maximum,value);
        }
        if (maximum==Float.NEGATIVE_INFINITY) { throw new IllegalArgumentException("Every token is banned"); }
        if (parameters.temperature()==0) { return new Choice(Sampler.argmax(logits,0,logits.length),0); }
        for (int i=0;i<working.length;i++) {
            // Shift in double before scaling, avoiding +infinity for finite logits and small temperatures.
            double value=((double)logits[i]-maximum)/parameters.temperature();
            working[i]=value < -Float.MAX_VALUE ? Float.NEGATIVE_INFINITY : (float)value;
        }
        int k=parameters.topK()==0 ? working.length : Math.min(parameters.topK(),working.length);
        Sampler.sample(working,0,working.length,1,k,1,0,probabilities,0,order,0);
        while (k>1 && probabilities[order[k-1]]==0) { k--; }
        double total=0;
        for (int i=0;i<k;i++) { total+=probabilities[order[i]]; }
        int kept=k;
        if (parameters.topP()<1) {
            double sum=0;
            for (int i=0;i<k;i++) {
                sum+=probabilities[order[i]];
                if (sum>=parameters.topP()*total) { kept=i+1; break; }
            }
        }
        double mass=0;
        for (int i=0;i<kept;i++) { mass+=probabilities[order[i]]; }
        // Vectrix 4.1's top-p threshold uses full-row mass. Select our normalized
        // top-k nucleus first, then ask Vectrix to sample exactly that ranked set.
        int token=Sampler.sample(working,0,working.length,1,kept,1,Philox.uniform(parameters.seed(),sampleIndex),probabilities,0,order,0);
        return new Choice(token,StrictMath.log(probabilities[token]/mass));
    }
}

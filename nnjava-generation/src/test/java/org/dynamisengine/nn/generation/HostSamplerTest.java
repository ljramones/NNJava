package org.dynamisengine.nn.generation;

import org.dynamisengine.vectrix.hash.Philox;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class HostSamplerTest {
    @Test void greedyUsesTheFirstMaximumAndPreservesInput() {
        float[] logits={Float.NEGATIVE_INFINITY,3,3,-2}; var sampler=new HostSampler(4,SamplingParameters.builder().temperature(0).define());
        assertEquals(new HostSampler.Choice(1,0),sampler.sample(logits,0)); assertArrayEquals(new float[]{Float.NEGATIVE_INFINITY,3,3,-2},logits);
    }

    @Test void topPIsRenormalizedWithinTopKAndEqualRanksPreferLowerIds() {
        var parameters=SamplingParameters.builder().topK(2).topP(.4f).seed(19).define(); var sampler=new HostSampler(5,parameters);
        for (int i=0;i<100;i++) { assertEquals(new HostSampler.Choice(0,0),sampler.sample(new float[5],i)); }
        var both=new HostSampler(5,parameters.toBuilder().topP(.75f).define());
        boolean first=false,second=false;
        for (int i=0;i<100;i++) {
            var value=both.sample(new float[5],i); first|=value.tokenId()==0; second|=value.tokenId()==1;
            assertTrue(value.tokenId()<2); assertEquals(-Math.log(2),value.logProbability(),1e-7);
        }
        assertTrue(first && second);
    }

    @Test void nucleusSelectionAndLogProbabilityMatchAnIndependentDistribution() {
        float[] logits={(float)Math.log(.6),(float)Math.log(.3),(float)Math.log(.1)};
        long seed=42; var sampler=new HostSampler(3,SamplingParameters.builder().topP(.7f).seed(seed).define());
        for (int i=0;i<1000;i++) {
            int expected=Philox.uniform(seed,i)<2.0/3 ? 0 : 1; var actual=sampler.sample(logits,i);
            assertEquals(expected,actual.tokenId()); assertEquals(Math.log(expected==0 ? 2.0/3 : 1.0/3),actual.logProbability(),1e-6);
        }
    }

    @Test void temperatureScalingMatchesKnownProbabilitiesAndIsStableForExtremeLogits() {
        var sampler=new HostSampler(2,SamplingParameters.builder().temperature(2).seed(7).define());
        for (int i=0;i<100;i++) {
            var actual=sampler.sample(new float[]{0,(float)(2*Math.log(3))},i);
            assertEquals(Math.log(actual.tokenId()==0 ? .25 : .75),actual.logProbability(),1e-6);
        }
        var extreme=new HostSampler(3,SamplingParameters.builder().temperature(Float.MIN_VALUE).define());
        assertEquals(new HostSampler.Choice(1,0),extreme.sample(new float[]{-Float.MAX_VALUE,Float.MAX_VALUE,Float.NEGATIVE_INFINITY},0));
    }

    @Test void samplingIsCounterBasedRatherThanDependentOnCallOrder() {
        var p=SamplingParameters.builder().seed(99).define(); var a=new HostSampler(3,p); var b=new HostSampler(3,p);
        float[] logits={0,.3f,.7f}; var expected=a.sample(logits,17); a.sample(logits,900);
        assertEquals(expected,a.sample(logits,17)); assertEquals(expected,b.sample(logits,17));
    }

    @Test void bannedTokensCannotBeSampledEvenWhenTheyHaveZeroProbabilityTies() {
        var sampler=new HostSampler(4,SamplingParameters.builder().seed(17).define());
        for (int i=0;i<100;i++) { assertEquals(new HostSampler.Choice(2,0),sampler.sample(new float[]{Float.NEGATIVE_INFINITY,-Float.MAX_VALUE,Float.MAX_VALUE,Float.NEGATIVE_INFINITY},i)); }
    }

    @Test void invalidSettingsLogitsAndCountersAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> SamplingParameters.builder().temperature(Float.NaN).define());
        assertThrows(IllegalArgumentException.class, () -> SamplingParameters.builder().topK(-1).define());
        assertThrows(IllegalArgumentException.class, () -> SamplingParameters.builder().topP(0).define());
        assertThrows(IllegalArgumentException.class, () -> SamplingParameters.builder().repetitionPenalty(0).define());
        var sampler=new HostSampler(2,SamplingParameters.builder().define());
        for (float[] logits:new float[][]{{Float.NaN,0},{0,Float.POSITIVE_INFINITY},{Float.NEGATIVE_INFINITY,Float.NEGATIVE_INFINITY},{1}}) {
            assertThrows(IllegalArgumentException.class, () -> sampler.sample(logits,0));
        }
        assertThrows(IllegalArgumentException.class, () -> sampler.sample(new float[]{0,1},-1));
    }
}

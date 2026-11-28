package org.dynamisengine.nn;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Proxy;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import org.dynamisengine.nn.backend.Backend;
import org.dynamisengine.nn.backend.vectrix.VectrixBackend;
import org.dynamisengine.nn.generation.*;
import org.dynamisengine.nn.io.ModelDefinitions;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tokenizers.BytePairTokenizer;
import org.dynamisengine.nn.transformer.CausalLanguageModel;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class GenerationIntegrationTest {
    @TempDir Path temporary;
    private static CausalLanguageModel.Definition definition() {
        return CausalLanguageModel.builder().vocabularySize(7).modelWidth(8).blockCount(2).contextLength(16)
                .attention(a -> a.queryHeads(2).keyValueHeads(1)).feedForward(f -> f.hiddenWidth(12)).seed(42).define();
    }

    @Test void cachedGreedyGenerationMatchesIndependentFullSequenceDecoding() {
        try (var context=ExecutionContext.cpu(); var model=definition().build(context)) {
            model.training(false); int before=context.liveAllocations(); var state=model.trainingState();
            try (var generation=GenerationSession.builder().model(model).maxNewTokens(4).sampling(s -> s.temperature(0)).build()) {
                var result=generation.generate(new int[]{0,1}); int[] history={0,1};
                for (int actual:result.tokenIds()) {
                    try (var input=context.tensor(history,Shape.of(1,history.length)); var logits=model.forward(input);
                         var last=Ops.slice(logits,1,history.length-1,history.length)) {
                        float[] values=last.toFloatArray(); int best=0;
                        for (int i=1;i<values.length;i++) { if (values[i]>values[best]) { best=i; } }
                        assertEquals(best,actual); history=Arrays.copyOf(history,history.length+1); history[history.length-1]=actual;
                    }
                }
                assertEquals(GenerationSession.StopReason.MAX_NEW_TOKENS,result.reason()); assertEquals(state,model.trainingState());
                assertEquals(before,context.liveAllocations());
            }
            assertEquals(before,context.liveAllocations()); assertFalse(model.isTraining());
        }
    }

    @Test void sampledIdsAndLogProbabilitiesMatchFullSequenceLogits() {
        var policy=SamplingParameters.builder().temperature(.7f).topK(3).topP(.9f).seed(11).define();
        try (var context=ExecutionContext.cpu(); var model=definition().build(context)) {
            model.training(false);
            try (var generation=GenerationSession.builder().model(model).maxNewTokens(4).sampling(policy).build()) {
                var result=generation.generate(new int[]{0,1}); int[] history={0,1}; var sampler=new HostSampler(7,policy);
                for (var actual:result.tokens()) {
                    try (var input=context.tensor(history,Shape.of(1,history.length)); var logits=model.forward(input);
                         var last=Ops.slice(logits,1,history.length-1,history.length)) {
                        var expected=sampler.sample(last.toFloatArray(),actual.index());
                        assertEquals(expected.tokenId(),actual.tokenId()); assertEquals(expected.logProbability(),actual.logProbability(),3e-6);
                        history=Arrays.copyOf(history,history.length+1); history[history.length-1]=actual.tokenId();
                    }
                }
                assertEquals(result.tokens(),generation.generate(new int[]{0,1}).tokens());
            }
        }
    }

    @Test void modeTapeAndBudgetErrorsLeaveTheModelAndContextUsable() {
        try (var context=ExecutionContext.cpu(); var model=definition().build(context);
             var generation=GenerationSession.builder().model(model).maxNewTokens(3).sampling(s -> s.temperature(0)).build()) {
            int before=context.liveAllocations();
            assertThrows(IllegalStateException.class, () -> generation.generate(new int[]{0,1})); assertTrue(model.isTraining());
            model.training(false);
            try (var tape=context.gradientTape()) { assertThrows(IllegalStateException.class, () -> generation.generate(new int[]{0,1})); }
            assertThrows(IllegalArgumentException.class, () -> generation.generate(new int[]{7}));
            assertThrows(IllegalArgumentException.class, () -> generation.generate(new int[14])); assertEquals(before,context.liveAllocations());
            assertEquals(3,generation.generate(new int[]{0,1}).tokens().size()); assertEquals(before,context.liveAllocations());
        }
    }

    @Test void callbackFailureAndStaleWeightsReleaseCachesBeforeTheNextRequest() {
        try (var context=ExecutionContext.cpu(); var model=definition().build(context)) {
            model.training(false); int before=context.liveAllocations();
            try (var generation=GenerationSession.builder().model(model).maxNewTokens(3).sampling(s -> s.temperature(0)).build()) {
                assertThrows(IllegalArgumentException.class, () -> generation.generate(new int[]{0},token -> { throw new IllegalArgumentException("Injected callback failure"); }));
                assertEquals(before,context.liveAllocations());
                assertThrows(IllegalStateException.class, () -> generation.generate(new int[]{0},token -> {
                    if (token.index()==0) {
                        try (var value=model.namedParameters().get("embedding.weight").value(); var zero=context.zeros(value.dtype(),value.shape())) { context.assign(value,zero); }
                    }
                    return GenerationSession.Decision.CONTINUE;
                }));
                assertEquals(before,context.liveAllocations());
                assertEquals(3,generation.generate(new int[]{0}).tokens().size()); assertEquals(before,context.liveAllocations());
            }
        }
    }

    @Test void failedCacheConstructionRollsBackEveryPartialCacheAllocation() {
        for (int failure=1;failure<=4;failure++) {
            var allocations=new AtomicInteger(); var failAt=new AtomicInteger(-1);
            try (var context=ExecutionContext.create(backend(allocations,failAt)); var model=definition().build(context)) {
                model.training(false); int before=context.liveAllocations();
                try (var generation=GenerationSession.builder().model(model).maxNewTokens(1).sampling(s -> s.temperature(0)).build()) {
                    assertEquals(before,context.liveAllocations()); failAt.set(allocations.get()+failure);
                    assertThrows(IllegalStateException.class, () -> generation.generate(new int[]{0})); assertEquals(before,context.liveAllocations());
                    assertEquals(1,generation.generate(new int[]{0}).tokens().size()); assertEquals(before,context.liveAllocations());
                }
            }
        }
    }

    @Test void generationSettingsPersistThroughTheNeutralDefinitionArchive() throws Exception {
        var config=GenerationSession.builder().maxNewTokens(4).minNewTokens(1).stopTokenIds(6).stopSequence(2,3)
                .sampling(s -> s.temperature(.7f).topK(3).topP(.9f).seed(19)).define();
        Path file=temporary.resolve("generation.nnjd"); ModelDefinitions.write(file,config.specification("generation"));
        var restored=GenerationConfig.fromSpec(ModelDefinitions.read(file)); assertEquals(config,restored);
        try (var context=ExecutionContext.cpu(); var model=definition().build(context)) {
            model.training(false);
            try (var a=GenerationSession.builder().model(model).configuration(config).build(); var b=GenerationSession.builder().model(model).configuration(restored).build()) {
                assertEquals(a.generate(new int[]{0,1}).tokens(),b.generate(new int[]{0,1}).tokens());
            }
        }
    }

    @Test void tokenGenerationComposesWithOptionalIncrementalUnicodeDecoding() {
        var tokenizer=BytePairTokenizer.builder().specialToken(256,"<bos>").specialToken(257,"<eos>").bosTokenId(256).eosTokenId(257).build();
        int[] script={0xc3,0xa9,0xf0,0x9f,0x98,0x80,257};
        AutoregressiveModel model=new AutoregressiveModel() {
            public int vocabularySize() { return tokenizer.vocabulary().size(); }
            public int contextLength() { return 16; }
            public Cursor openSession() {
                return new Cursor() {
                    int length,step;
                    public int length() { return length; }
                    public float[] prefill(int[] ids) { length=ids.length; return logits(); }
                    public float[] decode(int id) { assertEquals(script[step],id); step++; length++; return logits(); }
                    public void close() { }
                    private float[] logits() { float[] values=new float[vocabularySize()]; Arrays.fill(values,-100); values[script[Math.min(step,script.length-1)]]=10; return values; }
                };
            }
        };
        var streamed=new StringBuilder();
        try (var generation=GenerationSession.builder().model(model).maxNewTokens(10).stopTokenIds(257).sampling(s -> s.temperature(0)).build(); var decoder=tokenizer.newDecoder()) {
            var result=generation.generate(new int[]{256},token -> { streamed.append(decoder.accept(token.tokenId())); return GenerationSession.Decision.CONTINUE; });
            streamed.append(decoder.finish()); assertEquals("é😀",streamed.toString()); assertArrayEquals(script,result.tokenIds());
            assertEquals(tokenizer.decode(result.tokenIds()),streamed.toString()); assertEquals(GenerationSession.StopReason.STOP_TOKEN,result.reason());
        }
    }

    private static Backend backend(AtomicInteger allocations,AtomicInteger failAt) {
        var delegate=new VectrixBackend();
        return (Backend)Proxy.newProxyInstance(Backend.class.getClassLoader(),new Class<?>[]{Backend.class},(proxy,method,args) -> {
            if (method.getName().equals("allocate") && allocations.incrementAndGet()==failAt.get()) { throw new IllegalStateException("Injected cache allocation failure"); }
            try { return method.invoke(delegate,args); } catch (InvocationTargetException error) { throw error.getCause(); }
        });
    }
}

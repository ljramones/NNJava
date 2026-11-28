package org.dynamisengine.nn.generation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import org.dynamisengine.nn.model.ComponentSpec;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class GenerationSessionTest {
    @Test void streamingCommitsEveryTokenIncludingTheStopBeforeTheCallback() {
        var model=new Fake(new float[]{0,4,1},new float[]{0,1,5}); var streamed=new ArrayList<Integer>();
        try (var session=greedy(model,4).stopTokenIds(2).build()) {
            var result=session.generate(new int[]{0},token -> {
                assertEquals(token.index()+2,model.activeLength); assertEquals(token.tokenId(),model.lastToken);
                streamed.add(token.tokenId()); return GenerationSession.Decision.CONTINUE;
            });
            assertArrayEquals(new int[]{1,2},result.tokenIds()); assertEquals(List.of(1,2),streamed);
            assertEquals(GenerationSession.StopReason.STOP_TOKEN,result.reason()); assertEquals(1,model.closed);
            assertEquals(3,model.lastLength); assertArrayEquals(new int[]{0,1,2},result.allTokenIds());
            assertTrue(result.totalNanos()>=result.prefillNanos()+result.decodeNanos());
            assertThrows(UnsupportedOperationException.class, () -> result.tokens().clear());
        }
    }

    @Test void eachRequestHasAnIndependentCacheAndReproducibleSamplingStream() {
        var model=new Fake(new float[]{0,.2f,.4f});
        try (var session=GenerationSession.builder().model(model).maxNewTokens(12).sampling(s -> s.seed(42)).build()) {
            var first=session.generate(new int[]{0,1}); var second=session.generate(new int[]{0,1});
            assertEquals(first.tokens(),second.tokens()); assertEquals(2,model.opened); assertEquals(2,model.closed); assertEquals(24,model.decoded);
        }
    }

    @Test void minimumLengthCannotBeOverriddenByAProcessorReenablingEos() {
        var model=new Fake(new float[]{0,5,4});
        try (var session=greedy(model,4).minNewTokens(2).stopTokenIds(1).processor((history,logits) -> logits[1]=100).build()) {
            var result=session.generate(new int[]{0}); assertArrayEquals(new int[]{2,2,1},result.tokenIds());
            assertEquals(GenerationSession.StopReason.STOP_TOKEN,result.reason());
        }
    }

    @Test void stopSequencesMatchOnlyTheContinuationAndIncludeTheirTokens() {
        var model=new Fake(new float[]{0,1,5},new float[]{0,5,1},new float[]{0,1,5});
        try (var session=greedy(model,5).stopSequence(1,2).build()) {
            var result=session.generate(new int[]{1}); assertArrayEquals(new int[]{2,1,2},result.tokenIds());
            assertEquals(GenerationSession.StopReason.STOP_SEQUENCE,result.reason());
        }
    }

    @Test void stopSequencesHonorTheMinimumGeneratedCount() {
        var model=new Fake(new float[]{0,1,5});
        try (var session=greedy(model,4).minNewTokens(2).stopSequence(2).build()) {
            var result=session.generate(new int[]{0}); assertEquals(2,result.tokens().size()); assertEquals(GenerationSession.StopReason.STOP_SEQUENCE,result.reason());
        }
    }

    @Test void penaltiesCountThePromptAndPreviouslyGeneratedTokensWithoutMutatingSourceLogits() {
        float[] source={4,3,2,-1}; var model=new Fake(source); var inspected=new ArrayList<float[]>();
        try (var session=greedy(model,3).sampling(s -> s.repetitionPenalty(2).presencePenalty(.25f).frequencyPenalty(.5f))
                .processor((history,logits) -> inspected.add(logits.clone())).build()) {
            assertArrayEquals(new int[]{1,2,0},session.generate(new int[]{0,0}).tokenIds());
            assertEquals(.75f,inspected.getFirst()[0]); assertArrayEquals(new float[]{4,3,2,-1},model.logits[0]); assertArrayEquals(source,model.logits[0]);
        }
        try (var session=greedy(new Fake(new float[]{-2,-1,-3}),1).sampling(s -> s.repetitionPenalty(2)).build()) {
            assertArrayEquals(new int[]{0},session.generate(new int[]{1}).tokenIds());
        }
    }

    @Test void customHooksSeeImmutableHistoryAndCanStopAfterACommittedToken() {
        var model=new Fake(new float[]{0,5,1}); var history=new ArrayList<TokenHistory>();
        try (var session=greedy(model,5).processor((before,logits) -> { history.add(before); int[] copy=before.tokens(); copy[0]=2; })
                .stopping(after -> after.generatedCount()==2).build()) {
            var result=session.generate(new int[]{0}); assertEquals(GenerationSession.StopReason.CRITERION,result.reason());
            assertEquals(2,result.tokens().size()); assertArrayEquals(new int[]{0},history.getFirst().tokens()); assertEquals(0,history.getFirst().generatedCount());
            assertArrayEquals(new int[]{0,1},history.get(1).tokens());
        }
    }

    @Test void callbackStopAndCancellationDoNotPerformAnExtraDecode() {
        var model=new Fake(new float[]{0,5,1});
        try (var session=greedy(model,8).build()) {
            var stopped=session.generate(new int[]{0},token -> token.index()==1 ? GenerationSession.Decision.STOP : GenerationSession.Decision.CONTINUE);
            assertEquals(2,model.decoded); assertEquals(GenerationSession.StopReason.CALLBACK,stopped.reason());
            var cancelled=session.generate(new int[]{0},token -> { session.cancel(); return GenerationSession.Decision.CONTINUE; });
            assertEquals(3,model.decoded); assertEquals(1,cancelled.tokens().size()); assertEquals(GenerationSession.StopReason.CANCELLED,cancelled.reason());
            assertEquals(2,model.closed);
        }
    }

    @Test void cancellationMayComeFromAnotherThreadAndIsConsumedByOneRequest() throws Exception {
        var model=new Fake(new float[]{0,5,1});
        try (var session=greedy(model,2).build(); var executor=java.util.concurrent.Executors.newSingleThreadExecutor()) {
            executor.submit(session::cancel).get(); var result=session.generate(new int[]{0});
            assertEquals(GenerationSession.StopReason.CANCELLED,result.reason()); assertEquals(0,model.opened);
            assertEquals(2,session.generate(new int[]{0}).tokens().size());
            executor.submit(() -> assertThrows(IllegalStateException.class, () -> session.generate(new int[]{0}))).get();
        }
    }

    @Test void callbacksCannotReenterOrCloseTheSession() {
        var model=new Fake(new float[]{0,5,1});
        try (var session=greedy(model,1).build()) {
            session.generate(new int[]{0},token -> {
                assertThrows(IllegalStateException.class, () -> session.generate(new int[]{0})); assertThrows(IllegalStateException.class,session::close);
                return GenerationSession.Decision.CONTINUE;
            });
            assertEquals(1,model.opened); assertEquals(1,model.closed);
        }
    }

    @Test void cancellationDuringLogitProcessingIsObservedBeforeTheNextDecode() {
        var model=new Fake(new float[]{0,5,1}); GenerationSession[] owner=new GenerationSession[1];
        try (var session=greedy(model,3).processor((history,logits) -> owner[0].cancel()).build()) {
            owner[0]=session; var result=session.generate(new int[]{0});
            assertEquals(GenerationSession.StopReason.CANCELLED,result.reason()); assertEquals(0,result.tokens().size());
            assertEquals(0,model.decoded); assertEquals(1,model.closed);
        }
    }

    @Test void callbackOrDecodeFailureClosesTheCursorAndTheNextRequestStartsFresh() {
        var model=new Fake(new float[]{0,5,1});
        try (var session=greedy(model,3).build()) {
            assertThrows(IllegalArgumentException.class, () -> session.generate(new int[]{0},token -> { throw new IllegalArgumentException("Injected callback failure"); }));
            assertEquals(1,model.closed);
            model.failDecodeAt=2; assertThrows(IllegalStateException.class, () -> session.generate(new int[]{0})); assertEquals(2,model.closed);
            model.failDecodeAt=-1; assertEquals(3,session.generate(new int[]{0}).tokens().size()); assertEquals(3,model.closed);
        }
    }

    @Test void badPromptsAndBudgetsAreRejectedBeforeOpeningAnyCursor() {
        var model=new Fake(new float[]{0,5,1}); model.capacity=4;
        try (var session=greedy(model,3).build()) {
            assertThrows(IllegalArgumentException.class, () -> session.generate(new int[0]));
            assertThrows(IllegalArgumentException.class, () -> session.generate(new int[]{-1}));
            assertThrows(IllegalArgumentException.class, () -> session.generate(new int[]{3}));
            assertThrows(IllegalArgumentException.class, () -> session.generate(new int[]{0,1}));
            assertEquals(0,model.opened);
        }
        assertThrows(IllegalArgumentException.class, () -> greedy(model,2).stopTokenIds(3).build());
        assertThrows(IllegalArgumentException.class, () -> greedy(model,2).minNewTokens(3).define());
    }

    @Test void zeroBudgetDoesNotAllocateACacheOrEvaluateTheModel() {
        var model=new Fake(new float[]{0,5,1});
        try (var session=greedy(model,0).build()) {
            var result=session.generate(new int[]{0}); assertEquals(GenerationSession.StopReason.MAX_NEW_TOKENS,result.reason());
            assertEquals(0,result.tokenIds().length); assertEquals(0,model.opened); assertEquals(0,result.prefillNanos());
        }
    }

    @Test void invalidProviderOutputsAndAllBannedDistributionsCloseOwnedCursors() {
        var model=new Fake(new float[]{0,5,1});
        try (var session=greedy(model,1).build()) {
            model.badPrefill=true; assertThrows(IllegalArgumentException.class, () -> session.generate(new int[]{0})); assertEquals(1,model.closed);
            model.badPrefill=false; model.badLength=true; assertThrows(IllegalStateException.class, () -> session.generate(new int[]{0})); assertEquals(2,model.closed);
        }
        var banned=new Fake(new float[]{Float.NEGATIVE_INFINITY,Float.NEGATIVE_INFINITY});
        try (var session=greedy(banned,1).build()) { assertThrows(IllegalArgumentException.class, () -> session.generate(new int[]{0})); assertEquals(1,banned.closed); }
        var nan=new Fake(new float[]{Float.NaN,1});
        try (var session=greedy(nan,1).build()) { assertThrows(IllegalArgumentException.class, () -> session.generate(new int[]{0})); assertEquals(1,nan.closed); }
    }

    @Test void configurationRoundTripsAndBuildersSnapshotTheirMutableInputs() {
        var builder=greedy(new Fake(new float[]{0,1,2}),3).sampling(s -> s.seed(9)).sampling(s -> s.topK(2)).stopTokenIds(2).stopSequence(1,2);
        var config=builder.define(); assertEquals(9,config.sampling().seed());
        assertEquals(config,GenerationConfig.fromSpec(config.specification("sampling")));
        try (var session=builder.build()) { builder.maxNewTokens(9).stopTokenIds(0); assertEquals(config,session.configuration()); }
        var spec=config.specification("sampling"); var attrs=new LinkedHashMap<>(spec.attributes()); attrs.put("unknown","true");
        assertThrows(IllegalArgumentException.class, () -> GenerationConfig.fromSpec(new ComponentSpec(spec.type(),1,spec.name(),attrs,spec.children())));
        assertThrows(IllegalArgumentException.class, () -> GenerationConfig.fromSpec(new ComponentSpec(spec.type(),99,spec.name(),spec.attributes(),spec.children())));
        assertThrows(UnsupportedOperationException.class, () -> config.stopSequences().getFirst().add(0));
    }

    @Test void sourceAndCallerCannotMutateTheRequestPromptAndClosureIsExplicit() {
        var model=new Fake(new float[]{0,5,1}); model.mutatePrompt=true; int[] prompt={0};
        var session=greedy(model,1).build();
        var result=session.generate(prompt,token -> { prompt[0]=2; return GenerationSession.Decision.CONTINUE; });
        assertEquals(List.of(0),result.prompt()); session.close(); session.close();
        assertThrows(IllegalStateException.class, () -> session.generate(new int[]{0})); assertEquals(1,model.closed);
    }

    private static GenerationSession.Builder greedy(Fake model,int budget) { return GenerationSession.builder().model(model).maxNewTokens(budget).sampling(p -> p.temperature(0)); }

    private static final class Fake implements AutoregressiveModel {
        final float[][] logits;
        int capacity=64,opened,closed,decoded,lastLength,activeLength,lastToken,failDecodeAt=-1;
        boolean badPrefill,badLength,mutatePrompt;
        Fake(float[]... values) { logits=Arrays.stream(values).map(float[]::clone).toArray(float[][]::new); }
        public int vocabularySize() { return logits[0].length; }
        public int contextLength() { return capacity; }
        public Cursor openSession() {
            opened++; activeLength=0;
            return new Cursor() {
                int length,step;
                boolean isClosed;
                public int length() { return length; }
                public float[] prefill(int[] tokens) {
                    length=tokens.length; activeLength=length;
                    if (mutatePrompt) { tokens[0]=2; }
                    return badPrefill ? new float[1] : logits[0].clone();
                }
                public float[] decode(int id) {
                    if (++step==failDecodeAt) { throw new IllegalStateException("Injected decode failure"); }
                    decoded++; lastToken=id; if (!badLength) { length++; } activeLength=length;
                    return logits[Math.min(step,logits.length-1)].clone();
                }
                public void close() { if (!isClosed) { closed++; lastLength=length; isClosed=true; } }
            };
        }
    }
}

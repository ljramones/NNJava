package org.dynamisengine.nn.integration;

import java.util.concurrent.atomic.AtomicBoolean;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.transformer.CausalLanguageModel;
import org.dynamisengine.nn.transformer.InferenceOptions;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ProviderPagedInferenceTest {
    @Test void cpuRejectsOptionalPagedAndRecordedPoliciesBeforeAllocatingSessionState() {
        try (var context = ExecutionContext.cpu(); var model = CausalLanguageModel.builder().vocabularySize(7).modelWidth(8)
                .blockCount(1).contextLength(8).attention(a -> a.queryHeads(2)).feedForward(f -> f.hiddenWidth(12)).build(context)) {
            model.training(false); int baseline = context.liveAllocations();
            assertThrows(UnsupportedOperationException.class, () -> model.newSession(1, InferenceOptions.pagedFp16()));
            assertThrows(UnsupportedOperationException.class, () -> model.newSession(1, InferenceOptions.pagedFp16().withRecordedDecode()));
            var called = new AtomicBoolean();
            assertThrows(UnsupportedOperationException.class, () -> context.record(() -> { called.set(true); return null; }));
            assertFalse(called.get()); assertFalse(context.supportsCommandRecording()); assertEquals(baseline, context.liveAllocations());
            assertThrows(IllegalArgumentException.class, () -> new InferenceOptions(InferenceOptions.CachePrecision.FP16_PAGED, false, 31));
            assertThrows(IllegalArgumentException.class, () -> InferenceOptions.defaults().withRecordedDecode());
            try (var session = model.newSession(1)) { assertEquals(InferenceOptions.defaults(), session.options()); }
            assertEquals(baseline, context.liveAllocations());
        }
    }
}

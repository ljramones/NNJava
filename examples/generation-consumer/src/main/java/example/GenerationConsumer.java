package example;

import java.util.Arrays;

import org.dynamisengine.nn.generation.GenerationConfig;
import org.dynamisengine.nn.generation.GenerationSession;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.transformer.CausalLanguageModel;

/**
 * Installed-artifact token generation with no tokenizer, training, data, or IO module.
 */
public final class GenerationConsumer {
    public static void main(String[] args) {
        var definition = CausalLanguageModel.builder().vocabularySize(7).modelWidth(8).blockCount(2).contextLength(12)
                .attention(a -> a.queryHeads(2).keyValueHeads(1)).feedForward(f -> f.hiddenWidth(12)).seed(42).define();
        var config = GenerationSession.builder().maxNewTokens(4).sampling(s -> s.temperature(.7f).topK(3).topP(.9f).seed(19)).define();
        if (!config.equals(GenerationConfig.fromSpec(config.specification("sampling")))) {
            throw new AssertionError("Generation schema round trip failed");
        }
        try (var context = ExecutionContext.cpu(); var model = definition.build(context)) {
            model.training(false);
            int before = context.liveAllocations();
            try (var session = GenerationSession.builder().model(model).configuration(config).build()) {
                var first = session.generate(new int[]{0, 1});
                var second = session.generate(new int[]{0, 1});
                if (!first.tokens().equals(second.tokens()) || first.tokens().size() != 4 || context.liveAllocations() != before) {
                    throw new AssertionError("Packaged generation/reproducibility/cache ownership failed");
                }
                int[] last = first.allTokenIds();
                try (var input = context.tensor(last, Shape.of(1, last.length)); var output = model.forward(input)) {
                    if (!output.shape().equals(Shape.of(1, 6, 7))) {
                        throw new AssertionError("Generation closed its borrowed model");
                    }
                }
                System.out.println("Generation-only JPMS consumer passed; token IDs: " + Arrays.toString(first.tokenIds()));
            }
        }
    }
}

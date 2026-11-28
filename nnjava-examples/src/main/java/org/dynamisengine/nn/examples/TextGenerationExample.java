package org.dynamisengine.nn.examples;

import java.util.Arrays;
import java.util.Map;
import org.dynamisengine.nn.data.DataLoader;
import org.dynamisengine.nn.data.TensorDataset;
import org.dynamisengine.nn.generation.GenerationSession;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tokenizers.BytePairTokenizer;
import org.dynamisengine.nn.tokenizers.EncodeOptions;
import org.dynamisengine.nn.training.Optimizers;
import org.dynamisengine.nn.training.Trainer;
import org.dynamisengine.nn.transformer.CausalLanguageModel;

/** Tiny text fixture: caller-configured BPE, causal training, greedy generation and incremental Unicode output. */
public final class TextGenerationExample {
    private TextGenerationExample() { }
    public static void main(String[] args) throws Exception {
        try (var tokenizer=BytePairTokenizer.builder().merge('h','i',256)
                .specialToken(257,"<bos>").specialToken(258,"<eos>").bosTokenId(257).eosTokenId(258).build()) {
            String text="hi 😀";
            int[] sequence=tokenizer.encode(text,EncodeOptions.plain().withBos().withEos());
            int[] inputIds=Arrays.copyOf(sequence,sequence.length-1),targetIds=Arrays.copyOfRange(sequence,1,sequence.length);
            var definition=CausalLanguageModel.builder().vocabularySize(tokenizer.vocabulary().size()).modelWidth(8).blockCount(1).contextLength(16)
                    .attention(a -> a.queryHeads(2).keyValueHeads(1)).feedForward(f -> f.hiddenWidth(16)).seed(42).define();
            try (var context=ExecutionContext.cpu(); var model=definition.build(context);
                 var input=context.tensor(inputIds,Shape.of(1,inputIds.length)); var target=context.tensor(targetIds,input.shape());
                 var data=new TensorDataset(Map.of("input",input,"target",target)); var loader=DataLoader.builder(context,data).batchSize(1).build();
                 var trainer=Trainer.builder().model(model).optimizer(Optimizers.adamW().learningRate(.03f).define()).clipGlobalNorm(1).build()) {
                double first=trainer.evaluate(loader).loss(); trainer.fit(loader,150); double last=trainer.evaluate(loader).loss();
                model.training(false); var streamed=new StringBuilder();
                try (var generation=GenerationSession.builder().model(model).maxNewTokens(12).stopTokenIds(258).sampling(s -> s.temperature(0)).build();
                     var decoder=tokenizer.newDecoder()) {
                    var result=generation.generate(new int[]{257},token -> {
                        streamed.append(decoder.accept(token.tokenId())); return GenerationSession.Decision.CONTINUE;
                    });
                    streamed.append(decoder.finish());
                    if (last>=first*.1 || !text.equals(streamed.toString()) || !text.equals(tokenizer.decode(result.tokenIds()))
                            || result.reason()!=GenerationSession.StopReason.STOP_TOKEN) { throw new AssertionError("Text generation fixture failed"); }
                    System.out.printf("Text generation CE: %.6f -> %.6f; streamed: %s; stop: %s%n",first,last,streamed,result.reason());
                }
            }
        }
    }
}

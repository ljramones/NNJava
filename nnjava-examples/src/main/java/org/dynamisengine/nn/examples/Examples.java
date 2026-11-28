package org.dynamisengine.nn.examples;

/**
 * Small public-API examples; no downloads or GPU initialization are required.
 */
public final class Examples {
    private Examples() {
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 0 || args[0].equals("--list")) {
            System.out.println("linear-regression  Train a dense regression model");
            System.out.println("transformer        Train a small caller-defined decoder");
            System.out.println("mlp-xor            Fluent MLP, training, definition and weight restore");
            System.out.println("causal-lm          Fluent decoder, training, restore and cached inference");
            System.out.println("autoencoder        Trainer, shuffled data, evaluation and checkpoint resume");
            System.out.println("cnn                Convolution/pooling image classifier, validation and resume");
            System.out.println("rnn                Vanilla recurrent short-sequence recall, validation and resume");
            System.out.println("lstm               LSTM delayed recall, validation and resume");
            System.out.println("gru                GRU delayed recall, validation and resume");
            System.out.println("text-generation    Byte-level BPE, tiny decoder training, token sampling and Unicode streaming");
            return;
        }
        switch (args[0]) {
            case "linear-regression" -> LinearRegressionExample.main(new String[0]);
            case "transformer" -> TransformerTrainingExample.main(new String[0]);
            case "mlp-xor" -> XorTrainingExample.main(new String[0]);
            case "causal-lm" -> CausalLanguageModelExample.main(new String[0]);
            case "autoencoder" -> AutoencoderTrainingExample.main(new String[0]);
            case "cnn" -> CnnTrainingExample.main(new String[0]);
            case "rnn" -> RecurrentTrainingExample.run(org.dynamisengine.nn.nn.RecurrentCell.Kind.RNN);
            case "lstm" -> RecurrentTrainingExample.run(org.dynamisengine.nn.nn.RecurrentCell.Kind.LSTM);
            case "gru" -> RecurrentTrainingExample.run(org.dynamisengine.nn.nn.RecurrentCell.Kind.GRU);
            case "text-generation" -> TextGenerationExample.main(new String[0]);
            default -> throw new IllegalArgumentException("Unknown example: " + args[0] + "; use --list");
        }
    }
}

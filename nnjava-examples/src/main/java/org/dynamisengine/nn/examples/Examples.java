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
            return;
        }
        switch (args[0]) {
            case "linear-regression" -> LinearRegressionExample.main(new String[0]);
            case "transformer" -> TransformerTrainingExample.main(new String[0]);
            case "mlp-xor" -> XorTrainingExample.main(new String[0]);
            case "causal-lm" -> CausalLanguageModelExample.main(new String[0]);
            default -> throw new IllegalArgumentException("Unknown example: " + args[0] + "; use --list");
        }
    }
}

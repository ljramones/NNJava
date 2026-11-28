package org.dynamisengine.nn.examples;

import org.junit.jupiter.api.Test;

/** These small fixtures exercise the runnable public-API workflows, including temporary-file cleanup. */
class ExamplesSmokeTest {
    @Test void cnnTrainsValidatesAndResumes() throws Exception { Examples.main(new String[] {"cnn"}); }
    @Test void autoencoderTrainsAndResumes() throws Exception { Examples.main(new String[] {"autoencoder"}); }
    @Test void fluentMlpTrainsAndRestores() throws Exception { Examples.main(new String[] {"mlp-xor"}); }
    @Test void fluentDecoderTrainsRestoresAndUsesItsCache() throws Exception { Examples.main(new String[] {"causal-lm"}); }
}

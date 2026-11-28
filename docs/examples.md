# Neural Network and LLM Examples Module

`nnjava-examples` is a first-class Maven module with regression and small
transformer training examples, fluent MLP and causal-decoder workflows, and a
Trainer-based autoencoder, CNN, and RNN/LSTM/GRU classifiers with checkpoint resume.
The broader architecture catalog below remains planned where its layers are absent. It serves as a learning
catalog, an API usability check, and end-to-end validation of the library.

Reusable layers and cells belong in the production modules. Complete example
architectures, task-specific data preparation, and training recipes belong here.
Production modules must not depend on examples.

Current runner entries include `linear-regression`, `transformer` (direct component
composition), `mlp-xor`, `causal-lm`, `autoencoder`, `cnn`, `rnn`, `lstm`, and `gru`. All have CPU smoke
tests. Fluent models cover definition/state restore; the decoder checks caching,
and autoencoder/CNN examples compare resumed and uninterrupted training. See the
[builder API guide](model-builders.md) and [CNN guide](cnn-guide.md).

## Architecture catalog and order

| ID    | Example                               | Demonstrated task                                                                                        | Required library support                                                        | Delivery wave |
|-------|---------------------------------------|----------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|---------------|
| EX-01 | Shared runner and example conventions | List/select examples, seed/configure runs, report metrics, save/reload.                                  | Module extraction and builder conventions.                                      | R1 foundation |
| EX-02 | MLP / feed-forward classifier         | XOR, tabular classification, and an optional digit-classification run.                                   | Dense layers, activations, sequential builder, Trainer.                         | R2/R3         |
| EX-03 | Autoencoder                           | Reconstruct small vectors or images through a bottleneck.                                                | Encoder/decoder composition, reconstruction loss, model IO.                     | R3/R4         |
| EX-04 | CNN / LeNet-style classifier          | Classify small images.                                                                                   | Conv2D, pooling, flatten/reshape, dense head, image batches.                    | R4            |
| EX-05 | Vanilla RNN                           | Sequence classification or short time-series prediction.                                                 | Recurrent cell/layer, hidden-state handling, sequence data.                     | R4            |
| EX-06 | LSTM                                  | Delayed recall or sequence forecasting with longer dependencies.                                         | LSTM cell/state, sequence masking, backpropagation through time.                | R4            |
| EX-07 | GRU                                   | A gated recurrent alternative on a comparable sequence task.                                             | GRU cell/state and recurrent training APIs.                                     | R4            |
| EX-08 | Residual CNN                          | Build a small ResNet-style network.                                                                      | CNN primitives, residual/branch builders, suitable normalization.               | R4 follow-on  |
| EX-09 | Small causal transformer              | Construct and inspect a configurable decoder; compare full and cached logits.                            | Attention, positions, gated blocks, tied embeddings, LM builder, caches.        | R5 L1         |
| EX-10 | Encoder/decoder sequence model        | A small sequence-to-sequence task.                                                                       | Encoder/decoder builders and cross-attention or recurrent composition.          | R5 L6         |
| EX-11 | Real pretrained-model inference       | Load distinct layer weights and output head, tokenize, prefill, sample, and decode with a growing cache. | Weight reader/mapping, tokenizer, generation sessions, selected backend.        | R5 L2         |
| EX-12 | LM pretraining                        | Train on a small token corpus; measure held-out perplexity and resume.                                   | Shifted/masked token data, Trainer, token-normalized accumulation, checkpoints. | R5 L3         |
| EX-13 | Supervised and LoRA fine-tuning       | Apply token loss masks, train selected parameters/adapters, save and reload.                             | Fine-tuning and LoRA helpers, frozen-base checks, adapter persistence.          | R5 L4         |

RNN, LSTM, and GRU examples should expose hidden/cell state behavior explicitly:
sequence boundaries, state reset, variable lengths, and optional truncated
backpropagation. A cell interface and a sequence-level convenience layer should
both be available so the examples teach composition as well as the short API.

CNN convolution/pooling and recurrent cells/sequences are implemented on CPU;
EX-04 through EX-07 have runnable training/restore examples. The
[recurrent guide](recurrent-guide.md) also covers masks, carried state, resets,
and truncation. Where a
numerical primitive is missing, record the Vectrix kernel requirement separately;
examples must not implement replacement numerical kernels.

The [LLM roadmap](llm-roadmap.md) defines the L1-L6 delivery gates. EX-11 is the
first real-model GPU inference task, independent of later LM training. It must
validate reference logits, sampling, and cache growth, and measure the complete
public-API path. Repeated synthetic layer weights are insufficient for this gate.

## Module structure and planned families

```text
nnjava-examples/
  pom.xml
  src/main/java/org/dynamisengine/nn/examples/
    runner/
    mlp/
    autoencoder/
    cnn/
    rnn/
    lstm/
    gru/
    transformer/
      inference/
      pretraining/
      finetuning/
  src/main/resources/
  src/test/java/
```

Depend on nn, transformer, training, data, model IO, and the CPU numerical backend
as each example needs them; LLM examples add generation and optional tokenizers.
GPU examples use an optional profile after the selected numerical stack is
integrated and validated. The main roadmap records current release targets.
Keep the standalone Maven consumer as a separate packaging check; reactor examples
alone do not prove an external project can consume the artifacts. EX-11 must also
run through an external consumer.

## Requirements for every example

- Supply a short runnable entry point and a fluent model definition that a user
  can copy into another application. Avoid internal classes and direct kernel calls.
- Document the architecture, input/output shapes, chosen loss/optimizer, and task.
- Include a deterministic small dataset or generated fixture for quick execution.
  Full datasets are optional, explicitly acquired, and documented with provenance.
- Trainable examples show training/evaluation, metrics, model save/load, and a
  repeatable seed. Demonstrate resume where recurrent/random state matters.
  Pretrained inference examples show loading, generation, cache/session lifecycle,
  correctness checks, and end-to-end timings without requiring a training run.
- Provide a fast CPU smoke test with task-specific sanity checks and numerical
  tolerances. Keep full training separate from ordinary CI.
- Explain ownership and mode/state transitions, including recurrent state and KV
  caches where applicable.
- Keep real pretrained weights and full corpora outside ordinary CI. Use small
  deterministic fixtures there; acquire large integration inputs explicitly with
  documented model/tokenizer versions and provenance.

The shared runner should list available examples and accept an example name,
configuration/seed, dataset location, and a short/full run choice. Independent
entry points can remain available for IDE use. Heavy datasets and pretrained
weights should not be required by a normal Maven build.

## Acceptance and extension

The module is useful when a new user can select an architecture, run a small
training task, inspect its definition and metrics, and save/reload the result using
public APIs. Each additional architecture must identify its supporting primitive
features and provide its own observable acceptance task. Extend this catalog as
requirements emerge; architectures beyond the initial waves do not block the
first modular release.

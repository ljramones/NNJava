# Token generation and optional tokenizers

`nnjava-generation` implements single-sequence generation, Vectrix-backed host
sampling, streaming callbacks, stopping, cancellation, and reusable settings.
`nnjava-tokenizers` supplies byte-level BPE, vocabulary/special-token metadata,
provider discovery, and incremental UTF-8 decoding. Both target JDK 25.

## Modules and dependencies

| Artifact | JPMS name | Dependencies |
| --- | --- | --- |
| `nnjava-generation` | `org.dynamisengine.nn.generation` | transformer/core/nn, Vectrix core/ml 4.1.0 for host sampling; no training, data, IO, tokenizer, or native dependency |
| `nnjava-tokenizers` | `org.dynamisengine.nn.tokenizers` | java.base only |

The BOM aligns both artifacts. The CPU starter includes generation. Tokenizers
remain an explicit optional dependency, including for applications using the
starter. Neither transformer nor core imports generation/tokenizers. Independent
JPMS consumers verify generation without tokenizer/training/data/IO modules and
tokenization without any model or numerical module.

## Generate token IDs

```java
model.training(false);
try (var session = GenerationSession.builder()
        .model(model)
        .sampling(s -> s.temperature(.8f).topK(40).topP(.95f).seed(42))
        .maxNewTokens(64).minNewTokens(2)
        .stopTokenIds(eosTokenId)
        .stopSequence(firstStopId, secondStopId)
        .build()) {
    var result = session.generate(promptIds, token -> {
        output.accept(token.tokenId());
        return GenerationSession.Decision.CONTINUE;
    });
    int[] continuation = result.tokenIds();
}
```

Imports come from `org.dynamisengine.nn.generation`. The causal-model adapter
uses the model's public cache API and borrows its execution context. Applications
can supply other models through `AutoregressiveModel`: immutable vocabulary/context
metadata and a fresh owned `Cursor` that returns host FP32 logits after prefill
and each decode. Cursor position must include every appended input.

The prompt must be nonempty and `prompt.length + maxNewTokens` must fit the model's
context. IDs, budgets, and stop settings are validated before opening a cache.
Each request opens and closes its own cursor, including on failure. Reusing a
generation object starts a new request and restarts its configured Philox stream;
it does not carry a conversation cache implicitly. A zero budget performs no model
work. Evaluation mode and absence of a gradient tape are required for execution;
generation never changes shared model modes.

Every emitted token is decoded into the cache **before** its callback, including
EOS and the final budget token. Results/streams include stop-token sequences;
callers decide whether to display them. Token stop patterns match only the generated
continuation, not the prompt. `minNewTokens` suppresses stop IDs until that many
tokens have already been generated and ignores stop sequences before the minimum
count. Callback cancellation/custom stopping may end a request earlier.

Return `Decision.STOP`, install `.stopping(history -> condition)`, or call
`session.cancel()` to stop cooperatively. Only `cancel()` is cross-thread. Requests
and callbacks run on the owner thread; callbacks must not reenter or close their
session. Cancellation is checked before model work and between completed calls;
it does not interrupt an accepted GPU/CPU operation. The flag is consumed by one
request. Exceptions close its cursor; another request creates a fresh one, while
a failed backend context may still need application recreation.

## Sampling policy and configuration

Defaults are temperature 1, top-k 0 (all), top-p 1, repetition penalty 1,
presence/frequency penalties 0, seed 0, budget 128, minimum 0, and no stop IDs.
Temperature 0 selects greedy decoding; ties prefer the lowest ID. Repetition
penalties count prompt and generated IDs: divide positive logits and multiply
negative ones, then subtract presence and count-weighted frequency penalties.
Custom `.processor((history, logits) -> ...)` hooks run afterward on a copied row.
Negative infinity bans a token. NaN, positive infinity, all-banned rows, and penalty
overflow are rejected; minimum-length stop bans apply after custom processors.

`HostSampler` uses Vectrix `Sampler` ranking/probabilities and `Philox.uniform`.
Top-k runs first; top-p retains the shortest prefix reaching p of the **renormalized
top-k mass**. The implementation composes Vectrix calls to obtain that policy.
Each returned log probability belongs to the processed, temperature-scaled,
filtered distribution; greedy decoding reports 0. It is not the raw model log
probability. Shifting logits before temperature scaling handles finite extreme
values without positive overflow. Probability underflow follows FP32 behavior.

`SamplingParameters` and `GenerationConfig` are immutable. Builder `define()`
captures standard settings without allocating a model/cache. Persist them through
the neutral definition archive:

```java
ModelDefinitions.write(path, config.specification("generation"));
var restored = GenerationConfig.fromSpec(ModelDefinitions.read(path));
```

Custom hooks and models remain application-owned and are not serialized. Results
contain immutable prompt/token lists, stop reason, and timings. Prefill timing
includes cursor creation; decode timing includes all appended generation tokens;
total timing includes policy/hooks and cursor cleanup. Applications should measure
their complete text/model workflow separately for throughput or first-token claims.

## Byte-level BPE and Unicode streaming

```java
try (var tokenizer = BytePairTokenizer.builder()
        .merge('h', 'i', 256)
        .specialToken(257, "<bos>").bosTokenId(257)
        .specialToken(258, "<eos>").eosTokenId(258)
        .build()) {
    int[] ids = tokenizer.encode("hi 😀", EncodeOptions.plain().withBos());
    try (var decoder = tokenizer.newDecoder()) {
        for (int id : ids) { text.append(decoder.accept(id)); }
        text.append(decoder.finish());
    }
}
```

Imports come from `org.dynamisengine.nn.tokenizers`. The default vocabulary maps
the 256 byte values to IDs 0–255. Supply `.byteTokenIds(...)` for another byte-ID
mapping, then ordered `.merge(leftId,rightId,resultId)` rules and special tokens.
All IDs must form one contiguous vocabulary. Merge components must be base bytes
or earlier results; duplicate IDs/byte pieces/pairs are rejected. Lowest merge rank
wins, with leftmost occurrences breaking ties. A vocabulary piece is not an excuse
to bypass the merge sequence. There is no unknown-byte token: the full alphabet
encodes every valid Unicode scalar string. Unpaired UTF-16 surrogates are rejected.

Special spellings are literal text by default. `EncodeOptions.allowSpecial(...)`
opts into exact spellings; BOS/EOS are added only when requested. Default decoding
skips special IDs; `new DecodeOptions(false)` renders them. No punctuation or
whitespace cleanup is implicit. `TextNormalization.NONE` is the default; NFC/NFKC
are explicit. Normalization applies to ordinary text, after special recognition.

Default pre-tokenization treats the full ordinary text as a piece. Custom
`PreTokenizer` implementations and `PreTokenizers.regex(...)` must cover every
character without empty pieces. `PreTokenizers.gpt2()` supplies a GPT-2-style
boundary regex using JDK Unicode classes; it does not load a pretrained tokenizer.

Incremental decoders hold incomplete UTF-8 across token boundaries, emit complete
characters when available, and replace malformed sequences as Java UTF-8 does.
`finish()` flushes incomplete trailing bytes; closing alone discards pending text.
Decoder state is thread-confined. Built-in tokenizers/vocabularies are immutable;
custom pre-tokenizers must be safe for the application's sharing policy.

Text streaming composes with token generation through a callback that appends
`decoder.accept(token.tokenId())`, followed by `decoder.finish()`. The generation
module never imports the tokenizer. Match token IDs, merges, normalization,
special metadata, and pre-tokenization to model weights; vocabulary size alone
does not establish compatibility. Applications continue to own chat/prompt formats.

`TokenizerProviders.available()` discovers `TokenizerProvider` services on the
classpath or module path. The included `utf8-bytes` provider accepts only the
optional `normalization` setting. Additional providers may load formats/assets;
unknown provider IDs/settings fail explicitly.

## Verification and remaining scope

```sh
mvn install
mvn -f nnjava-examples/pom.xml exec:java -Dexec.args=text-generation
mvn -f examples/pom.xml verify
python3 tools/verify_module_consumers.py
```

The example trains a tiny decoder for `hi 😀`, generates token IDs through EOS,
and verifies exact streamed/whole-text decoding. Tests compare BPE against an
independent rescan oracle and sampling against known distributions, then compare
cached generation with full-model execution and verify lifecycle failures.

This chunk supplies single-sequence host sampling and configurable byte-level BPE.
GPU-resident sampling, batching/prefix sharing, beam/speculative decoding, SentencePiece,
and pretrained tokenizer/weight-file readers remain separate work. The existing
GGUF/Metal experiment retains its model-specific tokenizer/runtime independently;
its Llama pre-tokenization tables/cleanup have not been silently generalized here.

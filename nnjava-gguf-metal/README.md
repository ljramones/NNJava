# NNJava GGUF Metal inference

This optional Java 25 module runs the Llama 3.1 8B Instruct Q4_K_M GGUF mapping on native M4 Max Metal. A model holds the 32 distinct layers and Q6_K vocabulary head. A session owns an advancing paged FP16 KV cache. The public API returns complete FP32 logits, accepts application-supplied token IDs, and offers a checkpoint tokenizer for text input. No training or test-harness classes are runtime dependencies.

The module is built independently of the current CPU Maven project. It currently uses the local Step 12 GPU runtime under distinct snapshot coordinates. The most recent public DynamisGPU 1.14.1 predates direct FP16 unpack. The Vectrix dependency follows the tested GPU runtime. To prepare the exact private runtime, first build the GPU checkout, including its native bridge:

```sh
cd "$DYNAMISGPU"
mvn -Dmaven.repo.local="$PRIVATE_M2" -Pmetal-native -pl dynamis-gpu-api,dynamis-gpu-metal,dynamis-gpu-primitives -am package -DskipTests
cd "$NNJAVA/nnjava-gguf-metal"
python3 scripts/local-runtime.py "$DYNAMISGPU" "$PRIVATE_M2"
mvn -Dmaven.repo.local="$PRIVATE_M2" package org.apache.maven.plugins:maven-dependency-plugin:3.8.1:build-classpath -DincludeScope=runtime -Dmdep.outputFile=target/runtime-cp.txt
```

Use a private Maven repository with the build plugins and Vectrix 4.0.0 available. The helper writes three locally built artifacts under `1.15.0-step12-SNAPSHOT`, preserving the coordinates of released artifacts. After packaging, run the public text example from this directory:

```sh
NNJAVA_CP="target/nnjava-gguf-metal-0.1.0-SNAPSHOT.jar:$(cat target/runtime-cp.txt)"
java --enable-native-access=ALL-UNNAMED -cp "$NNJAVA_CP" examples/Generate.java "$GGUF" "The largest planet is"
```

## Model and session contract

`LlamaModel.open(path, capacity)` selects the packed-weight route. RoPE frequency factors, matrix dimensions, file type, tokenizer family, tensor count and expected quantized matrices are validated before opening Metal. Repacking preserves the original GGML codes and block parameters. The GGUF is memory-mapped and embeddings are dequantized on demand. The file backing the model must remain immutable for the model's lifetime.

`open(path, capacity, PrefillPrecision.ALLOW_FP16_AT_512)` permits weight reconstruction to FP16 only for 512-row groups, finite-half-range-qualified layer matrices and the model’s reusable scratch. It can change rounding compared with the packed path. The vocabulary projection stays packed. If the 112 MiB scratch cannot be allocated, the model falls back to packed projection. `reconstructionWeightCount()` reports how many weights can use the mode. Inputs, KV values and projection outputs inside the layers are FP16 in both modes.

Use `model.session()` to begin an empty sequence. A single live session per model owns the model scratch, and operations serialize on that model. Open additional model instances for fully independent sessions and memory budgets. Close a session to release its KV pages and reuse resident weights. Model closure invalidates an active session. Calls reject invalid IDs, capacity overflow, and invalid output offsets before recording GPU work. An exception after recording invalidates the model to prevent reuse of partial state.

- `session.prefill(ids)` appends input IDs and returns logits for the final input.
- `session.decode(id)` advances one token and returns all 128,256 logits.
- `session.append(ids, offsets...)` selects the zero-based rows of this input for which to return logits. Rows are increasing and an empty selection updates only the KV cache.
- `session.truncate(depth)` drops a suffix while retaining earlier KV pages.
- `session.generate(promptIds, maxNewTokens, temperature, seed, extraStops...)` samples on the host from complete FP32 logits. Temperature zero selects greedy decoding. The three native EOG IDs (end-of-text 128001, end-of-message 128008 and end-of-turn 128009) and explicit stop IDs end generation. The returned continuation and session position include the stop. The full requested prompt and token budget must fit the capacity.
- `model.tokenizer()` lazily loads checkpoint vocabulary and merges. Supply a caller-selected chat template if the application needs one.
- `model.loadTimes()` reports GGUF open and validation, matrix repacking, range checks, buffer upload including its wait, and total ready time. Some initialization, including pipeline compilation, is lazy; time the first token separately.

## Numerical evidence and limits

The original checkpoint sequence passed all nine frozen native logit checks in both modes. On three independent documents Step 12 selected 45 positions covering close decisions and decode after the reconstructed prefix. Packed and reconstructed NNJava each passed 43 of 45 native checks. Their worst KL divergences were 0.02174 and 0.02296 nats, respectively, against the declared 0.01 bound. Reconstructed and packed outputs passed all 45 pairwise tolerances; one top-token decision differed. The public default remains packed. Both routes retain the model-level limitations shown by these independent checks.

The production artifact contains the model, GPU graph, GGUF reader, range checker and tokenizer. The graph and reader were extracted from the previously verified real-checkpoint harness; profiling, frozen inputs and experimental server code stay outside the jar. This optional module does not alter the existing CPU build or introduce a training dependency. Performance figures are specific to the measured 40-core M4 Max.

## Hardware comparison through the packaged API

`examples/CheckpointRun.java` is an ordinary consumer of the installed public API. `scripts/checkpoint.py` compiles that consumer against the packaged JAR, checks the model SHA-256, and runs it under the inherited hardware benchmark lock. It records load and first-token timings, warm-file prompt ingestion, 128 advancing outputs, and greedy/seeded generation with EOS and stop checks. It verifies all frozen reference checks before timing, repeats sampling on the same seed, and races two overlapping session acquisitions to verify that exactly one acquires ownership. It checks close, reuse and prefix truncation. The timed trajectories retain the same occupied 4K or 32K position between runs.

```sh
python3 "$DYNAMISGPU/scripts/with-benchmark-lock.py" -- python3 scripts/checkpoint.py \
  --model "$GGUF" --tokens "$TOKEN_CSV" --depth 4096 --precision direct \
  --reference "$NATIVE_LOGITS" --runner-reference "$MATCHED_NNJAVA_LOGITS" \
  --cycles 4 --seconds 0 --out "$NEW_RESULT_DIRECTORY"
```

Choose a matching 4,224- or 32,896-row token CSV. Oracle captures use the naming scheme `logits-{position}.f32` with little-endian FP32 values. The native check covers prefix end and decode offsets 0, 63, 64 and 127, plus the four early 4K positions when those reference vectors exist. An optional same-precision NNJava capture must match bit for bit. A positive minimum duration extends the number of cycles. Sampled continuations are logged separately from the fixed-count, teacher-forced timing. Machine state and context depth affect throughput.

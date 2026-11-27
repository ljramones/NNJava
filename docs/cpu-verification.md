# CPU Framework Verification

The expanded CPU implementation was verified on 2026-11-23 using Temurin
25.0.4+7, Maven 3.10.2, macOS 27.0.1 arm64, and Vectrix 3.0.0. The plain and
SIMD-classifier configurations were run separately.

## Correctness and artifact checks

| Check | Command | Observed result |
| --- | --- | --- |
| Clean CPU build/install | `mvn -o clean install` | 54 tests passed; no failures, errors, or skips. Library JAR and POM installed locally. |
| SIMD execution | `mvn -o -q -Psimd test` | The same 54 tests passed with the incubator Vector API enabled. |
| Independent Maven consumer | `mvn -o -q -f examples/consumer/pom.xml clean compile exec:java` | Consumer transformer cross-entropy decreased from 1.591084 to 0.000009; token accuracy 1.000. |
| Benchmark packaging | `mvn -o -q -Pbenchmarks -DskipTests package` | JMH executable JAR built successfully. |
| Public API documentation | `mvn -o -q javadoc:jar` | Javadoc artifact generated successfully. |

Offline mode used already-installed dependencies and does not change test
semantics. CI is configured to build pinned parent/Vectrix source dependencies,
verify scalar/SIMD configurations, and run the independent consumer. Hosted CI
execution has not been observed in this local verification.

Tests cover independent numerical references, first-order finite differences,
batched/broadcast gradients, repeated indices, views/slices, masked attention,
normalization, positional transforms, optimizer fixtures, and nonlinear training.
A checkpoint test verifies eight subsequent CPU training steps and final weights
match bit for bit after restoring AdamW moments, dropout state, and loader order.
Additional tests reject incompatible checkpoint configuration and corrupted files.
Cached attention agrees with full-sequence execution for batched chunked prefill.

The consumer and training fixtures are small correctness examples. Their losses
and accuracy demonstrate API composition and differentiation, not production
model quality or large-model throughput.

## Initial JMH baseline

```sh
mvn -Pbenchmarks -DskipTests package
java -jar target/benchmarks.jar CoreBenchmark \
  -p features=16 -p batch=4 -wi 3 -i 5 -w 1s -r 1s -f 1 \
  -prof gc -rf json -rff target/cpu-baseline.json
```

JMH 1.37 used one fork, one thread, three one-second warmup iterations, and five
one-second measured iterations. This baseline uses plain Vectrix 3.0.0, with no
additional JVM arguments. [Raw JMH output](evidence/cpu-jmh.json) preserves samples,
confidence intervals, allocation measurements, JVM metadata, and parameters.

| Workload | Mean time | JMH error at 99.9 percent | Allocated bytes per operation |
| --- | ---: | ---: | ---: |
| Matrix multiplication and host readback | 1.126 microseconds | 0.089 microseconds | 2,896 |
| Linear forward, MSE backward, finite checks, and SGD update | 21.778 microseconds | 1.453 microseconds | 68,393 |

Inputs are `[4,16]`; matrix weights are `[16,16]`. Matrix measurement includes
output ownership, allocation, and a host copy. The training measurement clears
gradients, constructs/releases the tape, reads loss, runs backward, checks
gradients, and updates parameter/momentum buffers. Its learning rate is zero to
keep the workload stable across repetitions.

These measurements establish repeatable benchmark entry points and expose current
allocation cost. They are not isolated kernel timings, a scalar/SIMD comparison,
or an optimization claim. Larger shapes, longer runs, additional hardware, worker
scheduling, and profiling should guide subsequent performance changes. GPU
performance requires the separate NNJava adapter and real hardware validation.

package org.dynamisengine.nn.inference.llama31metal;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Objects;
import java.util.SplittableRandom;
import org.dynamisengine.gpu.api.compute.GpuDevice;
import org.dynamisengine.gpu.metal.MetalProvider;
import org.dynamisengine.gpu.primitives.PagedKvCache;

/**
 * Resident Llama 3.1 8B Q4_K_M weights for native Metal. One synchronous session owns the KV
 * cache and reusable scratch at a time. Closing the session releases its pages for the next session.
 * The caller owns all returned FP32 logits. Text and token-ID callers use the same inference path.
 */
public final class LlamaModel implements AutoCloseable {
    public enum PrefillPrecision { PACKED, ALLOW_FP16_AT_512 }
    /** Wall clock nanoseconds. The phase counters are contained in totalReadyNs. */
    public record LoadTimes(long totalReadyNs, long ggufOpenAndValidationNs, long matrixRepackNs,
                            long uploadAndWaitNs, long reconstructionRangeNs) {}

    private final Gguf file;
    private final GpuDevice device;
    private final MetalGraph graph;
    private final int capacity;
    private final LoadTimes times;
    private Session active;
    private boolean closed;
    private LlamaTokenizer tokenizer;

    public static LlamaModel open(Path gguf, int capacity) throws IOException {
        return open(gguf, capacity, PrefillPrecision.PACKED);
    }

    public static LlamaModel open(Path gguf, int capacity, PrefillPrecision precision) throws IOException {
        Objects.requireNonNull(precision);
        long start = System.nanoTime();
        var file = Gguf.open(gguf);
        GpuDevice device = null;
        try {
            validate(file, capacity);
            long openNs = System.nanoTime() - start;
            var adapter = new MetalProvider().adapters().stream().filter(a -> a.name().equals("Apple M4 Max"))
                    .findFirst().orElseThrow(() -> new UnsupportedOperationException("This adapter requires native Apple M4 Max Metal"));
            device = adapter.open();
            boolean allow = precision == PrefillPrecision.ALLOW_FP16_AT_512;
            var graph = new MetalGraph(device, file, capacity, allow);
            graph.prefillReconstruction(allow);
            var times = new LoadTimes(System.nanoTime()-start, openNs, graph.repackNs(), graph.uploadNs(), graph.rangeNs());
            return new LlamaModel(file, device, graph, capacity, times);
        } catch (RuntimeException | Error failure) {
            try {
                if (device != null) device.close();
            } catch (RuntimeException | Error closing) {
                failure.addSuppressed(closing);
            } finally {
                file.close();
            }
            throw failure;
        }
    }

    private LlamaModel(Gguf file, GpuDevice device, MetalGraph graph, int capacity, LoadTimes times) {
        this.file = file;
        this.device = device;
        this.graph = graph;
        this.capacity = capacity;
        this.times = times;
    }

    static void validate(Gguf f, int capacity) {
        int width = MetalGraph.D, ffn = MetalGraph.F, vocab = MetalGraph.VOCAB;
        if (!f.text("general.architecture").equals("llama") || !f.text("general.type").equals("model")
                || f.integer("general.file_type") != 15 || f.integer("general.quantization_version") != 2
                || f.integer("llama.block_count") != 32 || f.integer("llama.embedding_length") != width
                || f.integer("llama.feed_forward_length") != ffn || f.integer("llama.attention.head_count") != 32
                || f.integer("llama.attention.head_count_kv") != 8 || f.integer("llama.rope.dimension_count") != 128
                || f.integer("llama.vocab_size") != vocab || f.tensorCount() != 292
                || !f.text("tokenizer.ggml.pre").equals("llama-bpe") || !f.text("tokenizer.ggml.model").equals("gpt2"))
            throw new IllegalArgumentException("Expected the supported Llama 3.1 8B Q4_K_M GGUF mapping");
        if (capacity <= 0 || capacity > f.integer("llama.context_length") || capacity > 131072)
            throw new IllegalArgumentException("Context capacity is outside the supported model range");
        if (f.integer("tokenizer.ggml.bos_token_id") != 128000 || f.integer("tokenizer.ggml.eos_token_id") != 128009)
            throw new IllegalArgumentException("Unexpected tokenizer BOS or EOS ID");
        f.checkLlamaSpecialTokens();
        double eps=f.decimal("llama.attention.layer_norm_rms_epsilon");
        if (!(eps>0 && Float.isFinite((float)eps)) || f.decimal("llama.rope.freq_base") != 500000.0)
            throw new IllegalArgumentException("Unsupported RoPE base or invalid RMSNorm epsilon");
        requireTensor(f,"token_embd.weight",vocab,width,2,12);
        requireTensor(f,"output.weight",vocab,width,2,14);
        requireTensor(f,"output_norm.weight",1,width,1,0);
        requireTensor(f,"rope_freqs.weight",1,64,1,0);
        for (float factor:f.floats("rope_freqs.weight"))
            if (!(factor>0 && Float.isFinite(factor))) throw new IllegalArgumentException("Invalid RoPE frequency factor");
        for (int i=0;i<32;i++) {
            String b="blk."+i+".";
            requireTensor(f,b+"attn_norm.weight",1,width,1,0);
            requireTensor(f,b+"ffn_norm.weight",1,width,1,0);
            requireTensor(f,b+"attn_q.weight",width,width,2,12);
            requireTensor(f,b+"attn_k.weight",1024,width,2,12);
            requireTensor(f,b+"attn_v.weight",1024,width,2,12,14);
            requireTensor(f,b+"attn_output.weight",width,width,2,12);
            requireTensor(f,b+"ffn_up.weight",ffn,width,2,12);
            requireTensor(f,b+"ffn_gate.weight",ffn,width,2,12);
            requireTensor(f,b+"ffn_down.weight",width,ffn,2,12,14);
        }
    }

    private static void requireTensor(Gguf f, String name, int rows, int cols, int rank, int... types) {
        var t=f.tensor(name);
        if (t.rows()!=rows || t.cols()!=cols || t.rank()!=rank || Arrays.stream(types).noneMatch(id -> id==t.typeId()))
            throw new IllegalArgumentException("Unsupported shape or data type for "+name);
    }

    public int vocabularySize() { return MetalGraph.VOCAB; }
    public int capacity() { return capacity; }
    public LoadTimes loadTimes() { return times; }
    /** Number of immutable weight matrices qualified for the optional reconstructed 512-row prefill. */
    public synchronized int reconstructionWeightCount() { requireOpen(); return graph.directWeights(); }
    public synchronized LlamaTokenizer tokenizer() {
        requireOpen();
        if (tokenizer == null) tokenizer = LlamaTokenizer.from(file);
        return tokenizer;
    }
    public synchronized Session session() {
        requireOpen();
        if (active != null) throw new IllegalStateException("Close the live session before opening another");
        active = new Session(graph.sequence());
        return active;
    }
    private void requireOpen() { if (closed) throw new IllegalStateException("Model is closed"); }

    @Override public synchronized void close() {
        if (closed) return;
        closed = true;
        try {
            if (active != null) active.close();
        } finally {
            try { graph.close(); }
            finally {
                try { device.close(); }
                finally { file.close(); }
            }
        }
    }

    /** Single-owner KV sequence. Concurrent method calls on the resident model serialize. */
    public final class Session implements AutoCloseable {
        private final PagedKvCache.Sequence sequence;
        private Session(PagedKvCache.Sequence sequence) { this.sequence=sequence; }
        private void usable() {
            requireOpen();
            if (active!=this) throw new IllegalStateException("Session is closed");
        }
        public int position() { synchronized (LlamaModel.this) { usable(); return sequence.tokens(); } }

        /** Discard tokens after retainedTokens and keep the existing prefix in GPU memory. */
        public void truncate(int retainedTokens) {
            synchronized (LlamaModel.this) {
                usable();
                if (retainedTokens < 0 || retainedTokens > sequence.tokens())
                    throw new IllegalArgumentException("Invalid retained prefix length");
                graph.truncate(sequence, retainedTokens);
            }
        }

        /** Append a nonempty token sequence and return the last token's complete next-token distribution as logits. */
        public float[] prefill(int[] ids) { return append(ids, ids.length-1)[0]; }

        public float[] decode(int tokenId) { return append(new int[]{tokenId},0)[0]; }

        /**
         * Append IDs and return full logits at the selected zero-based offsets into this input.
         * The offsets must be strictly increasing. An empty selection advances KV without computing the head.
         */
        public float[][] append(int[] input, int... outputOffsets) {
            synchronized (LlamaModel.this) {
                usable();
                int[] ids=input.clone(), rows=outputOffsets.clone();
                checkRequest(ids, rows, capacity-sequence.tokens());
                float[][] outputs=new float[rows.length][];
                int next=0;
                try {
                    for (int offset=0;offset<ids.length;offset+=512) {
                        int count=Math.min(512,ids.length-offset);
                        int first=next;
                        while(next<rows.length && rows[next]<offset+count) next++;
                        int[] local=new int[next-first];
                        for (int j=0;j<local.length;j++) local[j]=rows[first+j]-offset;
                        var result=graph.append(sequence,ids,offset,count,local);
                        System.arraycopy(result,0,outputs,first,local.length);
                    }
                    return outputs;
                } catch (RuntimeException | Error failure) {
                    // A failed command can leave the GPU or reserved KV pages partially advanced.
                    try { LlamaModel.this.close(); }
                    catch (RuntimeException | Error closing) { failure.addSuppressed(closing); }
                    throw failure;
                }
            }
        }

        /**
         * Append the prompt and generate up to maxNewTokens, including a sampled EOS in the returned IDs and KV.
         * A zero temperature is greedy. A positive temperature samples the complete CPU-side logit vector using this seed.
         * The checkpoint end-of-text, end-of-message, and end-of-turn tokens always stop, and applications may supply additional stop token IDs.
         */
        public int[] generate(int[] prompt, int maxNewTokens, double temperature, long seed, int... extraStops) {
            synchronized (LlamaModel.this) {
                usable();
                if (maxNewTokens<0 || maxNewTokens>(long)capacity-sequence.tokens()-prompt.length)
                    throw new IllegalArgumentException("Generation exceeds remaining context");
                if (!Double.isFinite(temperature) || temperature<0) throw new IllegalArgumentException("Invalid temperature");
                int[] stops=extraStops.clone();
                for(int token:stops) checkId(token);
                Arrays.sort(stops);
                var rng=new SplittableRandom(seed);
                float[] logits=prefill(prompt);
                int[] tokens=new int[maxNewTokens];
                for (int i=0;i<tokens.length;i++) {
                    int token=sample(logits,temperature,rng);
                    tokens[i]=token;
                    logits=decode(token);
                    if(isEndOfGeneration(token) || Arrays.binarySearch(stops,token)>=0) return Arrays.copyOf(tokens,i+1);
                }
                return tokens;
            }
        }

        @Override public void close() {
            synchronized (LlamaModel.this) {
                if(active!=this) return;
                try { sequence.close(); } finally { active=null; }
            }
        }
    }

    static void checkRequest(int[] ids,int[] rows,int remaining) {
        if (ids.length==0 || ids.length>remaining) throw new IllegalArgumentException("Invalid token count for this session");
        for (int id:ids) checkId(id);
        int previous=-1;
        for (int row:rows) {
            if(row<=previous || row>=ids.length) throw new IllegalArgumentException("Invalid output row "+row);
            previous=row;
        }
    }
    /** Native end-of-text, end-of-message and end-of-turn for the metadata-validated checkpoint vocabulary. */
    public static boolean isEndOfGeneration(int tokenId) {
        return tokenId==128001 || tokenId==128008 || tokenId==128009;
    }

    private static void checkId(int id) {
        if(id<0 || id>=MetalGraph.VOCAB) throw new IllegalArgumentException("Invalid token ID "+id);
    }
    static int sample(float[] logits,double temperature,SplittableRandom rng) {
        int best=0;
        for(int i=0;i<logits.length;i++) {
            if (!Float.isFinite(logits[i])) throw new IllegalArgumentException("Nonfinite logit at row "+i);
            if(logits[i]>logits[best]) best=i;
        }
        if(temperature==0) return best;
        double sum=0;
        for(float value:logits) sum+=Math.exp(((double)value-logits[best])/temperature);
        double threshold=rng.nextDouble()*sum;
        double cumulative=0;
        for(int i=0;i<logits.length;i++) {
            cumulative+=Math.exp(((double)logits[i]-logits[best])/temperature);
            if(threshold<cumulative) return i;
        }
        return best;
    }
}

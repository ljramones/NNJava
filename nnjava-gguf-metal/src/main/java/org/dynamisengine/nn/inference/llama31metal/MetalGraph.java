package org.dynamisengine.nn.inference.llama31metal;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.time.Duration;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import org.dynamisengine.gpu.api.compute.*;
import org.dynamisengine.gpu.api.compute.kit.StridedView;
import org.dynamisengine.gpu.primitives.*;
import org.dynamisengine.gpu.primitives.GemmF16.Epilogue;

/** A synchronous, checked Llama 3.1 8B GGUF inference session for the public single-sequence Metal adapter. */
final class MetalGraph implements AutoCloseable {
    static final int D = 4096, H = 32, KV = 8, HD = 128, KD = KV * HD, F = 14336, VOCAB = 128256;
    private static final int PAGE = 64, BATCH = 512;
    private static final Duration TIMEOUT = Duration.ofMinutes(4);
    private final GpuDevice device;
    private final Gguf file;
    private final Norms norms;
    private final Rope rope;
    private final Cast cast;
    private final QuantizedGemm quant, headGemm;
    private final AttentionF16 attention;
    private final PagedKvCache cache;
    private final Rope.Table ropeTable;
    private final List<GpuBuffer> buffers = new ArrayList<>();
    private final Layer[] layers = new Layer[32];
    private final Weight head;
    private final float epsilon;
    private final GpuBuffer freq, finalNorm, logits, x0, x1, norm, q, k, v, qr, kr, heads, attnOutput, tokenOutput, mid, up, gate, positions;
    private long repackNs, uploadNs, rangeNs;
    private final boolean allowReconstruction;
    private int eligibleProjections;
    private boolean directPrefill;
    private GpuBuffer directScratch;
    private Quantize unpack;
    private GemmF16 dense;

    /** Set between requests on this single-owner synchronous session. Zero signals exact fallback. */
    public int prefillReconstruction(boolean requested) {
        if (!requested) {
            directPrefill = false;
            return 0;
        }
        if (!allowReconstruction) throw new IllegalStateException("FP16 weight reconstruction was not enabled at load");
        if (eligibleProjections == 0) return 0;
        if (directScratch == null) {
            try {
                directScratch = buffer(2L * F * D);
            } catch (ComputeException failure) {
                if (failure.code() != ComputeException.Code.RESOURCE_EXHAUSTED) throw failure;
                return 0;
            }
            unpack = Quantize.create(device);
            dense = GemmF16.create(device, GemmF16.Input.FP16, Precision.FP16);
        }
        directPrefill = true;
        return eligibleProjections;
    }

    private record Weight(QuantizedMatrix shape, GpuBuffer data, boolean safeHalf) {}
    private record Layer(GpuBuffer n1, GpuBuffer n2, Weight q, Weight k, Weight v, Weight o, Weight up, Weight gate, Weight down) {}
    long repackNs() { return repackNs; }
    long uploadNs() { return uploadNs; }
    long rangeNs() { return rangeNs; }
    int directWeights() { return directPrefill ? eligibleProjections : 0; }

    /** Open with an externally owned device; device closure also cleans up a partially loaded model. */
    MetalGraph(GpuDevice device, Gguf file, int capacity, boolean allowReconstruction) {
        this.device = device;
        this.file = file;
        this.allowReconstruction = allowReconstruction;
        if (!file.text("general.architecture").equals("llama") || file.integer("llama.block_count") != layers.length
                || file.integer("llama.embedding_length") != D || file.integer("llama.feed_forward_length") != F
                || file.integer("llama.attention.head_count") != H || file.integer("llama.attention.head_count_kv") != KV
                || file.integer("llama.rope.dimension_count") != HD || capacity < 1 || capacity > file.integer("llama.context_length"))
            throw new IllegalArgumentException("Expected a Llama 3.1 8B checkpoint and a supported context capacity");
        epsilon = (float) file.decimal("llama.attention.layer_norm_rms_epsilon");
        norms = Norms.create(device, Precision.FP16, Precision.FP16);
        rope = Rope.create(device, Precision.FP16, Precision.FP16);
        cast = Cast.create(device);
        quant = QuantizedGemm.create(device, GemmF16.Input.FP16, Precision.FP16);
        headGemm = QuantizedGemm.create(device, GemmF16.Input.FP16, Precision.FP32);
        attention = AttentionF16.create(device, Precision.FP16);
        cache = PagedKvCache.create(device, new PagedKvCache.Config(layers.length, KV, HD, PAGE, Math.ceilDiv(capacity, PAGE)));
        float[] factors = file.floats("rope_freqs.weight");
        if (factors.length != HD / 2) throw new IllegalArgumentException("RoPE factor count");
        double theta = file.decimal("llama.rope.freq_base");
        float[] cs = new float[capacity * HD];
        for (int i=0; i<factors.length; i++) {
            if (!(factors[i]>0 && Float.isFinite(factors[i]))) throw new IllegalArgumentException("Invalid RoPE factor");
            double frequency = Math.pow(theta, -2.0 * i / HD) / factors[i];
            for (int p=0; p<capacity; p++) {
                cs[p*HD+2*i]=(float)Math.cos(p*frequency);
                cs[p*HD+2*i+1]=(float)Math.sin(p*frequency);
            }
        }
        ropeTable = Rope.Table.of(capacity, HD, cs);
        freq = upload(ropeTable.bytes());
        for (int i=0; i<layers.length; i++) {
            String b="blk."+i+".";
            layers[i]=new Layer(vector(b+"attn_norm.weight"), vector(b+"ffn_norm.weight"),
                    weight(b+"attn_q.weight",D,D), weight(b+"attn_k.weight",KD,D), weight(b+"attn_v.weight",KD,D),
                    weight(b+"attn_output.weight",D,D), weight(b+"ffn_up.weight",F,D), weight(b+"ffn_gate.weight",F,D),
                    weight(b+"ffn_down.weight",D,F));
        }
        finalNorm=vector("output_norm.weight");
        head=weight("output.weight",VOCAB,D);
        logits=buffer(4L*VOCAB);
        positions=buffer(4L*BATCH);
        x0=activation(D); x1=activation(D); norm=activation(D);
        q=activation(D); k=activation(KD); v=activation(KD); qr=activation(D); kr=activation(KD);
        heads=activation(D); attnOutput=activation(D); tokenOutput=activation(D); mid=activation(D);
        up=activation(F); gate=activation(F);
    }

    private GpuBuffer activation(int width) { return buffer(2L*BATCH*width); }
    private GpuBuffer buffer(long size) {
        var b=device.allocate(new BufferDescriptor(size,EnumSet.of(BufferUsage.STORAGE,BufferUsage.TRANSFER_SOURCE,BufferUsage.TRANSFER_DESTINATION),Placement.PRIVATE));
        buffers.add(b);
        return b;
    }
    private GpuBuffer upload(byte[] bytes) {
        long start = System.nanoTime();
        var b=buffer(bytes.length);
        try (var e=device.encoder()) {
            e.upload(b,bytes);
            try (var s=device.queue().submit(e)) { await(s); }
        }
        uploadNs += System.nanoTime() - start;
        return b;
    }
    private GpuBuffer vector(String name) {
        float[] data=file.floats(name);
        if (data.length!=D) throw new IllegalArgumentException(name+" width");
        var bytes=ByteBuffer.allocate(4*D).order(ByteOrder.LITTLE_ENDIAN);
        bytes.asFloatBuffer().put(data);
        return upload(bytes.array());
    }
    private Weight weight(String name,int rows,int cols) {
        var desc=file.tensor(name);
        if (desc.rows()!=rows || desc.cols()!=cols) throw new IllegalArgumentException(name+" shape");
        long start = System.nanoTime();
        var packed=file.matrix(name);
        repackNs += System.nanoTime() - start;
        start = System.nanoTime();
        boolean safeHalf = allowReconstruction && !name.equals("output.weight")
                && ReconstructionRange.withinFiniteHalf(packed.matrix(), packed.bytes());
        rangeNs += System.nanoTime() - start;
        if (safeHalf) eligibleProjections++;
        return new Weight(packed.matrix(),upload(packed.bytes()),safeHalf);
    }
    private static void await(GpuSubmission s) {
        if (!s.await(TIMEOUT)) throw new IllegalStateException("Metal submission timed out");
    }
    PagedKvCache.Sequence sequence() { return cache.newSequence(); }
    void truncate(PagedKvCache.Sequence sequence, int tokens) { cache.truncate(sequence, tokens); }

    private void project(GpuCommandEncoder e, GpuBuffer in, Weight w, GpuBuffer out, int m, Epilogue ep, GpuBuffer g, GpuBuffer residual) {
        var shape=GemmF16.Shape.ofTransposedB(m,w.shape.rows(),w.shape.cols(),1,0);
        if (m == BATCH && directPrefill && w != head && w.safeHalf()) {
            unpack.unpackF16(e, w.shape(), w.data(), directScratch);
            dense.encode(e, in, directScratch, out, shape, ep, null, g, residual);
            return;
        }
        (w==head ? headGemm : quant).encode(e,in,w.shape,w.data,out,shape,ep,null,g,residual);
    }
    private void layer(GpuCommandEncoder e, PagedKvCache.Sequence seq, int i, int start,int m,GpuBuffer in,GpuBuffer out) {
        Layer b=layers[i];
        norms.encode(e,Norms.Kind.RMS,in,null,null,b.n1,null,norm,Norms.Shape.of(m,D,epsilon));
        project(e,norm,b.q,q,m,Epilogue.NONE,null,null);
        project(e,norm,b.k,k,m,Epilogue.NONE,null,null);
        project(e,norm,b.v,v,m,Epilogue.NONE,null,null);
        // GGUF Q and K rows are already permuted to the GGML adjacent-pair convention.
        rope.encode(e,q,positions,ropeTable,freq,qr,Rope.Shape.of(m,H,HD,Rope.Layout.INTERLEAVED));
        rope.encode(e,k,positions,ropeTable,freq,kr,Rope.Shape.of(m,KV,HD,Rope.Layout.INTERLEAVED));
        cache.write(e,i,seq,start,kr,v,m,HD,KD);
        var shape=new AttentionF16.Shape(H,m,start+m,HD,(float)(1/Math.sqrt(HD)),true,KV,start+m);
        if (m==1) {
            attention.encode(e,qr,cache,i,seq,tokenOutput,shape);
        } else {
            var headView=StridedView.of(0,new long[]{H,m,HD},new long[]{HD,D,1});
            var headDest=StridedView.of(0,new long[]{H,m,HD},new long[]{(long)m*HD,HD,1});
            cast.cast(e,Cast.Type.FP16,Cast.Type.FP16,qr,headView,heads,headDest);
            attention.encode(e,heads,cache,i,seq,attnOutput,shape);
            var tokenView=StridedView.of(0,new long[]{m,H,HD},new long[]{HD,(long)m*HD,1});
            var tokenDest=StridedView.of(0,new long[]{m,H,HD},new long[]{D,HD,1});
            cast.cast(e,Cast.Type.FP16,Cast.Type.FP16,attnOutput,tokenView,tokenOutput,tokenDest);
        }
        project(e,tokenOutput,b.o,mid,m,Epilogue.NONE.withResidual(),null,in);
        norms.encode(e,Norms.Kind.RMS,mid,null,null,b.n2,null,norm,Norms.Shape.of(m,D,epsilon));
        project(e,norm,b.up,up,m,Epilogue.NONE,null,null);
        project(e,norm,b.gate,gate,m,Epilogue.NONE.withActivation(Epilogue.Activation.SILU).withGate(),up,null);
        project(e,gate,b.down,out,m,Epilogue.NONE.withResidual(),null,mid);
    }

    /** Validated input and output rows are relative to this caller's array and this batch respectively. */
    float[][] append(PagedKvCache.Sequence seq, int[] ids, int offset, int count, int[] outputs) {
        int start = seq.tokens();
        var input = ByteBuffer.allocate(2 * D * count).order(ByteOrder.LITTLE_ENDIAN);
        var pos = ByteBuffer.allocate(4 * count).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < count; i++) {
            for (float value : file.embedding(ids[offset+i])) input.putShort(Float.floatToFloat16(value));
            pos.putInt(start+i);
        }
        var result = new float[outputs.length][];
        try (var e = device.encoder()) {
            try (var dst = x0.slice(0, input.capacity()); var p = positions.slice(0, pos.capacity())) {
                e.upload(dst, input.array()).upload(p, pos.array());
            }
            cache.reserve(e, seq, count);
            for (int i=0; i<layers.length; i++) layer(e, seq, i, start, count, i%2==0 ? x0 : x1, i%2==0 ? x1 : x0);
            var reads = new GpuReadback[outputs.length];
            for (int j=0; j<outputs.length; j++) {
                try (var row = x0.slice(2L * D * outputs[j], 2L * D)) {
                    norms.encode(e, Norms.Kind.RMS, row, null, null, finalNorm, null, norm, Norms.Shape.of(1,D,epsilon));
                }
                project(e, norm, head, logits, 1, Epilogue.NONE, null, null);
                reads[j] = e.download(logits);
            }
            try (var submission = device.queue().submit(e)) {
                await(submission);
                for (int j=0; j<reads.length; j++) {
                    result[j] = new float[VOCAB];
                    ByteBuffer.wrap(reads[j].bytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(result[j]);
                }
            }
        }
        return result;
    }
    @Override public void close() {
        if (dense != null) dense.close();
        if (unpack != null) unpack.close();
        cache.close(); attention.close(); quant.close(); headGemm.close(); cast.close(); rope.close(); norms.close();
        for (var b:buffers.reversed()) b.close();
    }
}

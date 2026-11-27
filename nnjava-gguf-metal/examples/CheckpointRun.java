import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import org.dynamisengine.nn.inference.llama31metal.LlamaModel;

/** Standalone consumer of the packaged public API; measured decode includes all FP32 logits on the host. */
public final class CheckpointRun {
    private static void require(boolean ok,String message) { if (!ok) throw new IllegalStateException(message); }

    public static void main(String[] args) throws Exception {
        if (args.length!=7) throw new IllegalArgumentException("GGUF tokens.csv depth cycles minimum-seconds packed|direct output");
        Path file=Path.of(args[0]);
        var csv=Files.readAllLines(Path.of(args[1]));
        require(csv.getFirst().equals("position,input_token"),"Invalid token inventory");
        int[] ids=new int[csv.size()-1];
        for(int i=0;i<ids.length;i++) {
            var fields=csv.get(i+1).split(",");
            require(Integer.parseInt(fields[0])==i,"Invalid token position");
            ids[i]=Integer.parseInt(fields[1]);
        }
        int depth=Integer.parseInt(args[2]), cycles=Integer.parseInt(args[3]);
        double duration=Double.parseDouble(args[4]);
        if(depth<1 || ids.length!=depth+128 || cycles<1 || !Double.isFinite(duration) || duration<0)
            throw new IllegalArgumentException("Expected a prefix plus 128 advancing tokens and valid timing requests");
        var precision=switch (args[5]) {
            case "direct" -> LlamaModel.PrefillPrecision.ALLOW_FP16_AT_512;
            case "packed" -> LlamaModel.PrefillPrecision.PACKED;
            default -> throw new IllegalArgumentException("Invalid prefill precision");
        };
        Path out=Files.createDirectories(Path.of(args[6]));
        int[] prefix=Arrays.copyOf(ids,depth);
        try (var model=LlamaModel.open(file,ids.length,precision)) {
            var load=model.loadTimes();
            try(var sessions=model.session()) {
                try {model.session(); throw new AssertionError("Concurrent session accepted");}
                catch(IllegalStateException expected) {require(sessions.position()==0,"Live sequence changed");}
                try {sessions.decode(-1); throw new AssertionError("Invalid token accepted");}
                catch(IllegalArgumentException expected) {require(sessions.position()==0,"Invalid token advanced sequence");}
                long first=System.nanoTime();
                float[] firstLogits=sessions.decode(ids[0]);
                long firstNs=System.nanoTime()-first;
                require(firstLogits.length==model.vocabularySize() && Float.isFinite(firstLogits[0]),"First token");
                String cold="total_ready_ns,gguf_open_validate_ns,repack_ns,upload_wait_ns,range_check_ns,first_token_ns,direct_weights\n"
                    +load.totalReadyNs()+","+load.ggufOpenAndValidationNs()+","+load.matrixRepackNs()+","+load.uploadAndWaitNs()+","
                    +load.reconstructionRangeNs()+","+firstNs+","+model.reconstructionWeightCount()+"\n";
                Files.writeString(out.resolve("startup.csv"),cold);
            }
            try (var seq=model.session();
                 var timings=Files.newBufferedWriter(out.resolve("trajectories.csv"));
                 var steps=Files.newBufferedWriter(out.resolve("steps.csv"))) {
                int[] selected=depth==4096?new int[]{0,63,64,511,depth-1}:new int[]{depth-1};
                // Measure the ordinary prefix with only the last head row, then capture any extra positions separately.
                long start=System.nanoTime();
                var last=seq.prefill(prefix);
                long prefixNs=System.nanoTime()-start;
                Files.writeString(out.resolve("prefill.csv"),"tokens,wall_ns\n"+depth+","+prefixNs+"\n");
                dump(out,depth-1,last);
                if(selected.length>1) {
                    seq.truncate(0);
                    var vectors=seq.append(prefix,selected);
                    for(int i=0;i<selected.length;i++) dump(out,selected[i],vectors[i]);
                    require(Arrays.equals(last,vectors[selected.length-1]),"Additional captured rows changed prefix-end logits");
                }
                for(int i=0;i<128;i++) {
                    float[] v=seq.decode(ids[depth+i]);
                    if(i==0 || i==63 || i==64 || i==127) dump(out,depth+i,v);
                }
                require(seq.position()==ids.length,"Advancing capture length");
                // Validate that keeping prefix pages across subsequent trajectories reproduces the captured decode exactly.
                seq.truncate(depth);
                float[] replay=seq.decode(ids[depth]);
                require(Arrays.equals(bytes(replay),Files.readAllBytes(out.resolve("logits-"+depth+".f32"))),"Retained prefix changed logits");
                timings.write("cycle,tokens,wall_ns\n");
                steps.write("cycle,position,wall_ns\n");
                long began=System.nanoTime();
                int c=0;
                do {
                    seq.truncate(depth);
                    long[] ts=new long[128];
                    long tick=System.nanoTime();
                    for(int i=0;i<128;i++) {
                        long b=System.nanoTime();
                        float[] v=seq.decode(ids[depth+i]);
                        require(v.length==model.vocabularySize() && Float.isFinite(v[0]),"Invalid logits");
                        ts[i]=System.nanoTime()-b;
                    }
                    long wall=System.nanoTime()-tick;
                    c++;
                    timings.write(c+",128,"+wall+"\n");
                    for(int i=0;i<128;i++) steps.write(c+","+(depth+i)+","+ts[i]+"\n");
                    timings.flush();steps.flush();
                } while(c<cycles || (System.nanoTime()-began)*1e-9<duration);
            }
            try (var seq=model.session()) {
                var tokenizer=model.tokenizer();
                String prompt="The largest planet in our solar system is";
                int[] promptIds=tokenizer.encode(prompt,true,false);
                int count=Math.min(8,model.capacity()-promptIds.length);
                int[] greedy=seq.generate(promptIds,count,0,117);
                seq.truncate(0);
                if (count>0) {
                    int[] stopped=seq.generate(promptIds,count,0,117,greedy[0]);
                    require(stopped.length==1 && seq.position()==promptIds.length+1,"Stop-token or position handling");
                    seq.truncate(0);
                }
                int[] sampled=seq.generate(promptIds,count,.7,117);
                seq.truncate(0);
                require(Arrays.equals(sampled,seq.generate(promptIds,count,.7,117)),"Seed roundtrip");
                Files.writeString(out.resolve("generation.txt"),"prompt="+prompt+"\ngreedy_tokens="+Arrays.toString(greedy)
                    +"\ngreedy_text="+tokenizer.decode(greedy,false)+"\nsampled_tokens="+Arrays.toString(sampled)
                    +"\nsampled_text="+tokenizer.decode(sampled,false)+"\nseed=117\ntemperature=0.7\n");
                seq.close();
                try { seq.decode(0); throw new AssertionError("Closed session accepted input"); }
                catch(IllegalStateException expected) { }
            }
        }
        Files.writeString(out.resolve("completed"),"ok\n");
    }

    private static byte[] bytes(float[] values) {
        var b=ByteBuffer.allocate(4*values.length).order(ByteOrder.LITTLE_ENDIAN);
        b.asFloatBuffer().put(values);
        return b.array();
    }
    private static void dump(Path out,int position,float[] values) throws Exception {
        for(float value:values) require(Float.isFinite(value),"Nonfinite logits at "+position);
        Files.write(out.resolve("logits-"+position+".f32"),bytes(values));
    }
}

import java.nio.file.Path;
import org.dynamisengine.nn.inference.llama31metal.LlamaModel;

/** java --enable-native-access=ALL-UNNAMED -cp classpath Generate.java /path/to/model.gguf "Your prompt" */
public final class Generate {
    public static void main(String[] args) throws Exception {
        if(args.length!=2) throw new IllegalArgumentException("GGUF path and prompt required");
        try(var model=LlamaModel.open(Path.of(args[0]),4096); var session=model.session()) {
            var tokenizer=model.tokenizer();
            int[] ids=tokenizer.encode(args[1],true,false);
            int[] reply=session.generate(ids,32,0.7,401);
            System.out.println(tokenizer.decode(reply,false));
        }
    }
}

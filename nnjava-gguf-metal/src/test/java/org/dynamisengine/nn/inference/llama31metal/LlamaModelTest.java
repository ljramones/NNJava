package org.dynamisengine.nn.inference.llama31metal;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class LlamaModelTest {
    @Test void rejectsInvalidInputsBeforeReservingKv() {
        assertDoesNotThrow(() -> LlamaModel.checkRequest(new int[]{0,128255},new int[]{0,1},2));
        assertDoesNotThrow(() -> LlamaModel.checkRequest(new int[]{1},new int[0],1));
        assertThrows(IllegalArgumentException.class, () -> LlamaModel.checkRequest(new int[0],new int[0],2));
        assertThrows(IllegalArgumentException.class, () -> LlamaModel.checkRequest(new int[]{1,2},new int[]{1},1));
        assertThrows(IllegalArgumentException.class, () -> LlamaModel.checkRequest(new int[]{-1},new int[0],2));
        assertThrows(IllegalArgumentException.class, () -> LlamaModel.checkRequest(new int[]{128256},new int[0],2));
        assertThrows(IllegalArgumentException.class, () -> LlamaModel.checkRequest(new int[]{1,2},new int[]{1,1},2));
        assertThrows(IllegalArgumentException.class, () -> LlamaModel.checkRequest(new int[]{1,2},new int[]{1,0},2));
        assertThrows(IllegalArgumentException.class, () -> LlamaModel.checkRequest(new int[]{1},new int[]{-1},2));
    }

    @Test void usesAllNativeStopsWithoutConstructingTokenizer() {
        for(int token:new int[]{128001,128008,128009}) assertTrue(LlamaModel.isEndOfGeneration(token));
        for(int token:new int[]{0,128000,128007,128010}) assertFalse(LlamaModel.isEndOfGeneration(token));
    }

    @Test void generationIsSeededAndGreedyBreaksTiesById() {
        float[] x={-1f,4f,4f,2.5f};
        assertEquals(1,LlamaModel.sample(x,0,new SplittableRandom(7)));
        var a=new SplittableRandom(17);
        var b=new SplittableRandom(17);
        for(int i=0;i<100;i++) assertEquals(LlamaModel.sample(x,.7,a),LlamaModel.sample(x,.7,b));
        assertEquals(2,LlamaModel.sample(new float[]{-1000,0,Float.MAX_VALUE},.00001,a));
        assertThrows(IllegalArgumentException.class, () -> LlamaModel.sample(new float[]{Float.NaN,1},1,b));
    }

    @Test void rejectsMalformedFileBeforeOpeningDevice(@TempDir Path tmp) throws Exception {
        Path file=tmp.resolve("invalid.gguf");
        Files.write(file,new byte[24]);
        assertThrows(IllegalArgumentException.class, () -> LlamaModel.open(file,8));
        Files.write(file,new byte[]{0x47,0x47,0x55,0x46});
        assertThrows(IllegalArgumentException.class, () -> LlamaModel.open(file,8));
        Files.delete(file);
    }
}

package org.dynamisengine.nn.inference.llama31metal;

import org.dynamisengine.gpu.primitives.QuantizedMatrix;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ReconstructionRangeTest {
    @Test void overflowingOrNonfiniteLastBlockUsesPackedPath() {
        for (var format : new QuantizedMatrix.Format[]{QuantizedMatrix.Format.Q4K, QuantizedMatrix.Format.Q6K}) {
            var weights = new QuantizedMatrix(format,256,2,512);
            byte[] data = new byte[Math.toIntExact(weights.bytes())];
            int end = Math.toIntExact(weights.parameterOffset()+weights.parameterBytes());
            int last = end - (format==QuantizedMatrix.Format.Q4K ? 16 : 20);
            for (float finite : new float[]{Float.float16ToFloat((short)1),-.01f,1f}) {
                putHalf(data,last,finite);
                assertTrue(ReconstructionRange.withinFiniteHalf(weights,data));
            }
            for (float excessive : new float[]{-65504f,Float.NaN,Float.POSITIVE_INFINITY,Float.NEGATIVE_INFINITY}) {
                putHalf(data,last,excessive);
                assertFalse(ReconstructionRange.withinFiniteHalf(weights,data));
            }
            if (format==QuantizedMatrix.Format.Q4K) {
                putHalf(data,last,0);
                putHalf(data,last+2,-65504f);
                assertFalse(ReconstructionRange.withinFiniteHalf(weights,data));
            }
        }
    }
    private static void putHalf(byte[] data,int offset,float v) {
        short h=Float.floatToFloat16(v);
        data[offset]=(byte)h;
        data[offset+1]=(byte)(h>>>8);
    }
}

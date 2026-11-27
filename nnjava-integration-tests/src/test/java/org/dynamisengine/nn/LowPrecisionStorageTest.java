package org.dynamisengine.nn;

import org.dynamisengine.nn.io.StateArchive;
import org.dynamisengine.nn.io.StateDict;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.state.TensorSnapshot;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class LowPrecisionStorageTest {
    @TempDir
    Path temporary;

    @Test
    void storagePreservesBitsThroughViewsCopiesAndHostSnapshots() {
        short[] bits = {0, (short) 0x8000, (short) 0x7e31, (short) 0xffff};
        for (DType dtype : new DType[]{DType.FP16, DType.BF16}) {
            try (var context = ExecutionContext.cpu();
                 var tensor = context.tensor(bits, dtype, Shape.of(2, 2));
                 var view = Ops.transpose(tensor);
                 var copy = Ops.copy(view)) {
                assertArrayEquals(new short[]{bits[0], bits[2], bits[1], bits[3]}, copy.toShortArray());
                var snapshot = TensorSnapshot.capture(copy);
                short[] exposed = snapshot.shorts();
                exposed[0] = 123;
                assertArrayEquals(copy.toShortArray(), snapshot.shorts());
                assertFalse(copy.requiresGrad());
                assertThrows(UnsupportedOperationException.class, () -> Ops.cast(copy, DType.FP32));
                assertArrayEquals(bits, tensor.toShortArray());
            }
        }
    }

    @Test
    void archivesRetainLowPrecisionStorageTypesAndPayloads() throws Exception {
        var shape = Shape.of(3);
        var fp16 = new TensorSnapshot(shape, DType.FP16, new short[]{1, (short) 0x8000, (short) 0x7e31});
        var bf16 = new TensorSnapshot(shape, DType.BF16, new short[]{2, (short) 0xffff, (short) 0x7fc1});
        var model = new StateDict(Map.of("fp16", fp16, "bf16", bf16), Map.of(), Map.of(), Map.of());
        Path path = temporary.resolve("storage.nnj");
        StateArchive.write(path, new StateArchive.Loaded(model, null, Map.of()));
        var restored = StateArchive.read(path).model();
        assertEquals(DType.FP16, restored.tensors().get("fp16").dtype());
        assertEquals(DType.BF16, restored.tensors().get("bf16").dtype());
        assertArrayEquals(fp16.shorts(), restored.tensors().get("fp16").shorts());
        assertArrayEquals(bf16.shorts(), restored.tensors().get("bf16").shorts());
    }
}

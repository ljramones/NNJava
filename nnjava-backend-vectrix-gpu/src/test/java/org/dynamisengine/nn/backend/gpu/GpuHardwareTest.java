package org.dynamisengine.nn.backend.gpu;

import static org.junit.jupiter.api.Assertions.*;

import org.dynamisengine.gpu.api.compute.GpuDevice;
import org.dynamisengine.gpu.api.compute.GpuProvider;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/**
 * Explicit hardware profiles fail on absent devices; ordinary builds need no native runtime.
 */
@EnabledIfSystemProperty(named = "nn.gpu.backend", matches = "METAL|VULKAN")
class GpuHardwareTest {
    private static GpuDevice device() throws Exception {
        String platform = System.getProperty("nn.gpu.backend");
        String type = platform.equals("METAL") ? "org.dynamisengine.gpu.metal.MetalProvider"
                : "org.dynamisengine.gpu.vulkan.VulkanProvider";
        var provider = (GpuProvider) Class.forName(type).getConstructor().newInstance();
        var adapters = provider.adapters();
        assertFalse(adapters.isEmpty(), "Requested GPU backend has no available device: " + platform);
        return adapters.getFirst().open();
    }

    @Test
    void exactTranscendentalsMatchTheCpuForNormalValues() throws Exception {
        float[] values = {0.125f, 0.5f, 1, 2, 3, 7};
        try (var cpu = ExecutionContext.cpu();
             var gpu = ExecutionContext.create(new GpuBackend(device(), GpuMode.EXACT));
             var reference = cpu.tensor(values, Shape.of(values.length));
             var input = gpu.tensor(values, Shape.of(values.length));
             var expectedExp = Ops.exp(reference); var actualExp = Ops.exp(input);
             var expectedLog = Ops.log(reference); var actualLog = Ops.log(input);
             var expectedSqrt = Ops.sqrt(reference); var actualSqrt = Ops.sqrt(input)) {
            assertArrayEquals(expectedExp.toFloatArray(), actualExp.toFloatArray());
            assertArrayEquals(expectedLog.toFloatArray(), actualLog.toFloatArray());
            assertArrayEquals(expectedSqrt.toFloatArray(), actualSqrt.toFloatArray());
        }
    }

    @Test
    void typedTransfersAndLowPrecisionStoragePreserveRawValues() throws Exception {
        for (GpuMode mode : GpuMode.values()) {
            try (var context = ExecutionContext.create(new GpuBackend(device(), mode));
                 var ints = context.tensor(new int[]{-1, 0, Integer.MAX_VALUE}, Shape.of(3));
                 var bools = context.tensor(new boolean[]{true, false, true}, Shape.of(3));
                 var empty = context.zeros(DType.FP32, Shape.of(0))) {
                assertArrayEquals(new int[]{-1, 0, Integer.MAX_VALUE}, ints.toIntArray());
                assertArrayEquals(new boolean[]{true, false, true}, bools.toBooleanArray());
                assertArrayEquals(new float[0], empty.toFloatArray());
                for (DType dtype : new DType[]{DType.FP16, DType.BF16}) {
                    short[] bits = {1, (short) 0x8000, (short) 0x7e31};
                    try (var storage = context.tensor(bits, dtype, Shape.of(3)); var copy = Ops.copy(storage)) {
                        assertArrayEquals(bits, copy.toShortArray());
                    }
                }
            }
        }
    }

    @Test
    void queuedArithmeticRetainsInputsAndViewsStayOnDevice() throws Exception {
        for (GpuMode mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var context = ExecutionContext.create(backend)) {
                var a = context.tensor(new float[]{1, 2, 3, 4, 5, 6}, Shape.of(2, 3));
                var result = Ops.square(a);
                a.close();
                try (result; var view = Ops.transpose(result); var copied = Ops.copy(view)) {
                    assertArrayEquals(new float[]{1, 16, 4, 25, 9, 36}, copied.toFloatArray());
                    assertArrayEquals(copied.toFloatArray(), view.toFloatArray());
                    assertThrows(UnsupportedOperationException.class, () -> Ops.sum(result));
                    assertArrayEquals(new float[]{1, 4, 9, 16, 25, 36}, result.toFloatArray());
                }
                assertEquals(0, context.liveAllocations());
                assertEquals(mode == GpuMode.EXACT ? "vectrix-gpu" : "dynamis-gpu-primitives",
                        backend.executedTiers().get(org.dynamisengine.nn.backend.Operation.MULTIPLY));
            }
        }
    }

    @Test
    void matrixProductsIncludeDecodeShapesAndTransposeViews() throws Exception {
        for (GpuMode mode : GpuMode.values()) {
            var backend = new GpuBackend(device(), mode);
            try (var context = ExecutionContext.create(backend);
                 var storedLeft = context.tensor(new float[]{99, 99, 99, 1, 2, 3, 4, 5, 6}, Shape.of(3, 3));
                 var left = Ops.slice(storedLeft, 0, 1, 3, 1);
                 var storedRight = context.tensor(new float[]{1, 2, 3, 4, 5, 6}, Shape.of(2, 3));
                 var right = Ops.transpose(storedRight);
                 var product = Ops.matmul(left, right);
                 var decode = context.tensor(new float[]{1, 2, 3}, Shape.of(1, 3));
                 var token = Ops.matmul(decode, right)) {
                assertArrayEquals(new float[]{14, 32, 32, 77}, product.toFloatArray());
                assertArrayEquals(new float[]{14, 32}, token.toFloatArray());
                assertEquals(mode == GpuMode.THROUGHPUT, backend.matmulAccuracy().isPresent());
            }
        }
    }

    @Test
    void throughputSupportsBroadcastAndExactModeReportsUnsupportedLayouts() throws Exception {
        for (GpuMode mode : GpuMode.values()) {
            try (var context = ExecutionContext.create(new GpuBackend(device(), mode));
                 var matrix = context.tensor(new float[]{1, 2, 3, 4, 5, 6}, Shape.of(2, 3));
                 var bias = context.tensor(new float[]{10, 20, 30}, Shape.of(3))) {
                if (mode == GpuMode.THROUGHPUT) {
                    try (var sum = Ops.add(matrix, bias)) {
                        assertArrayEquals(new float[]{11, 22, 33, 14, 25, 36}, sum.toFloatArray());
                    }
                } else {
                    assertThrows(UnsupportedOperationException.class, () -> Ops.add(matrix, bias));
                    assertArrayEquals(new float[]{1, 2, 3, 4, 5, 6}, matrix.toFloatArray());
                }
            }
        }
    }
}

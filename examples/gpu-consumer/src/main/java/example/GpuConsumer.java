package example;

import java.util.ServiceLoader;

import org.dynamisengine.nn.backend.BackendProvider;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.Shape;

/**
 * Verifies provider discovery on the module path with an explicitly supplied Metal runtime.
 */
public final class GpuConsumer {
    public static void main(String[] args) {
        String id = args.length == 0 ? "vectrix-gpu-exact" : args[0];
        var provider = ServiceLoader.load(BackendProvider.class).stream().map(ServiceLoader.Provider::get)
                .filter(p -> p.id().equals(id)).findFirst().orElseThrow();
        var devices = provider.devices();
        if (devices.isEmpty()) {
            throw new IllegalStateException("Requested native GPU runtime has no device");
        }
        try (var context = ExecutionContext.open(id, devices.getFirst().id());
             var a = context.tensor(new float[]{1, 2, 3, 4}, Shape.of(2, 2));
             var b = context.tensor(new float[]{5, 6, 7, 8}, Shape.of(2, 2));
             var result = Ops.matmul(a, b)) {
            if (!java.util.Arrays.equals(new float[]{19, 22, 43, 50}, result.toFloatArray())) {
                throw new AssertionError("GPU matrix product differs");
            }
            neuralWorkflow(context);
            attentionWorkflow(context);
            System.out.println("GPU module-path consumer passed: " + context.device());
        }
    }

    private static void neuralWorkflow(ExecutionContext context) {
        int before = context.liveAllocations();
        try (var input = context.tensor(new float[]{1, .2f, -.3f, -.5f, .7f, 1}, Shape.of(2, 3), true);
             var weights = context.tensor(new float[]{.2f, .5f, -.1f, -.3f, .4f, .7f}, Shape.of(3, 2), true);
             var scale = context.tensor(new float[]{1, 1}, Shape.of(2), true);
             var bias = context.tensor(new float[]{.1f, -.2f}, Shape.of(2), true);
             var target = context.tensor(new float[]{.2f, .8f, .7f, .3f}, Shape.of(2, 2));
             var tape = context.gradientTape(); var projected = Ops.matmul(input, weights);
             var shifted = Ops.add(projected, bias); var active = Ops.silu(shifted);
             var normalized = Ops.rmsNorm(active, scale, 1e-5f); var probabilities = Ops.softmax(normalized, -1);
             var difference = Ops.subtract(probabilities, target); var squared = Ops.square(difference); var loss = Ops.mean(squared)) {
            tape.backward(loss);
            float[] values = probabilities.toFloatArray();
            for (int row = 0; row < 2; row++) {
                if (Math.abs(values[row * 2] + values[row * 2 + 1] - 1) > 2e-5f) { throw new AssertionError("GPU probability row does not sum to one"); }
            }
            for (var tensor : new org.dynamisengine.nn.tensor.Tensor[]{input, weights, scale, bias}) {
                try (var gradient = tensor.gradient()) {
                    for (float value : gradient.toFloatArray()) { if (!Float.isFinite(value)) { throw new AssertionError("GPU neural gradient is not finite"); } }
                }
            }
            try (var gradient = weights.gradient()) { context.axpy(weights, gradient, -.01f); }
        }
        context.synchronize();
        if (context.liveAllocations() != before) { throw new AssertionError("GPU neural workflow leaked tensor allocations"); }
    }

    private static void attentionWorkflow(ExecutionContext context) {
        int before = context.liveAllocations();
        try (var table = context.tensor(new float[]{.1f, .2f, .3f, .4f, -.2f, .5f, .1f, -.3f, .7f, -.4f, .2f, .6f, -.1f, .3f, .8f, .2f}, Shape.of(4, 4), true);
             var rawIds = context.tensor(new int[]{-1, 1, 2}, Shape.of(3)); var zeroId = context.tensor(new int[]{0}, Shape.of());
             var valid = Ops.greater(rawIds, zeroId); var ids = Ops.where(valid, rawIds, zeroId);
             var positionTable = context.tensor(new int[]{0, 2, 4, 6}, Shape.of(4)); var positions = Ops.gather(positionTable, 0, ids);
             var allowed = context.tensor(new boolean[]{true, false, false, true, true, false, true, true, true}, Shape.of(3, 3));
             var excluded = context.full(Shape.of(), Float.NEGATIVE_INFINITY); var tape = context.gradientTape();
             var embedding = Ops.gather(table, 0, ids); var heads = Ops.reshape(embedding, Shape.of(1, 1, 3, 4));
             var query = Ops.rope(heads, positions, 10000, true); var key = Ops.transpose(query);
             var product = Ops.matmul(query, key); var scores = Ops.scale(product, .5f); var masked = Ops.where(allowed, scores, excluded);
             var probabilities = Ops.softmax(masked, -1); var output = Ops.matmul(probabilities, heads);
             var square = Ops.square(output); var loss = Ops.mean(square)) {
            tape.backward(loss);
            try (var gradient = table.gradient(); var finite = Ops.isFinite(gradient); var all = Ops.all(finite); var next = Ops.argmax(output, -1, false)) {
                if (!all.toBooleanArray()[0]) { throw new AssertionError("GPU attention embedding gradient is not finite"); }
                if (next.toIntArray().length != 3) { throw new AssertionError("GPU argmax output shape differs"); }
            }
            float[] values = probabilities.toFloatArray();
            if (values[0] != 1 || values[1] != 0 || values[2] != 0) { throw new AssertionError("GPU causal attention mask differs"); }
        }
        context.synchronize();
        if (context.liveAllocations() != before) { throw new AssertionError("GPU attention workflow leaked tensor allocations"); }
    }
}

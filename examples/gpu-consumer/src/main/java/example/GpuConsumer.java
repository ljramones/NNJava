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
            System.out.println("GPU module-path consumer passed: " + context.device());
        }
    }
}

package org.dynamisengine.nn.backend.gpu;

import java.util.List;
import java.util.ArrayList;
import java.util.ServiceLoader;

import org.dynamisengine.gpu.api.compute.GpuAdapter;
import org.dynamisengine.gpu.api.compute.GpuProvider;
import org.dynamisengine.nn.backend.Backend;
import org.dynamisengine.nn.runtime.Device;

final class GpuProviders {
    private GpuProviders() {
    }

    static List<GpuAdapter> adapters(String platform) {
        List<GpuProvider> providers = new ArrayList<>(ServiceLoader.load(GpuProvider.class).stream()
                .map(ServiceLoader.Provider::get).toList());
        // The 1.14.1 native jars export constructors but do not register GpuProvider
        // services. Load only explicitly installed/resolved platform modules.
        for (String type : List.of("org.dynamisengine.gpu.metal.MetalProvider",
                "org.dynamisengine.gpu.vulkan.VulkanProvider")) {
            if (platform != null && !type.toUpperCase(java.util.Locale.ROOT).contains("." + platform + ".")) {
                continue;
            }
            if (providers.stream().anyMatch(p -> p.getClass().getName().equals(type))) {
                continue;
            }
            try {
                var provider = (GpuProvider) Class.forName(type).getConstructor().newInstance();
                providers.add(provider);
            } catch (ClassNotFoundException absent) {
                // Native providers are optional; absence is reported by create().
            } catch (ReflectiveOperationException error) {
                throw new IllegalStateException("Could not create installed GPU provider " + type, error);
            }
        }
        return providers.stream().filter(p -> platform == null || p.backend().name().equals(platform))
                .flatMap(p -> p.adapters().stream()).toList();
    }

    static String key(GpuAdapter adapter) {
        return adapter.backend().name() + ":" + adapter.id();
    }

    static List<Device> devices(String provider) {
        return adapters(null).stream().map(a -> new Device(provider, key(a))).toList();
    }

    static Backend create(String key, GpuMode mode) {
        String platform = key.contains(":") ? key.substring(0, key.indexOf(':')) : key;
        var matches = adapters(platform).stream().filter(a -> key(a).equals(key)).toList();
        if (matches.size() != 1) {
            throw new IllegalArgumentException("GPU device is unavailable or ambiguous: " + key
                    + ". Supply a native runtime module, or construct GpuBackend with an explicit GpuDevice.");
        }
        return new GpuBackend(matches.getFirst().open(), mode);
    }
}

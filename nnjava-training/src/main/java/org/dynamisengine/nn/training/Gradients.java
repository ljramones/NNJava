package org.dynamisengine.nn.training;

import org.dynamisengine.nn.nn.Parameter;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.tensor.Tensor;

import java.util.*;

public final class Gradients {
    private Gradients() {
    }

    public static void requireFinite(Tensor gradient) {
        try (Tensor finite = Ops.isFinite(gradient); Tensor all = Ops.all(finite)) {
            if (!all.toBooleanArray()[0]) {
                throw new ArithmeticException("Nonfinite gradient; optimizer update aborted");
            }
        }
    }

    /**
     * Return the unclipped global norm and scale each unique gradient once.
     */
    public static double clipGlobalNorm(Collection<Parameter> parameters, double maximum) {
        if (!Double.isFinite(maximum) || maximum <= 0) {
            throw new IllegalArgumentException("Positive finite gradient norm required");
        }
        Set<Long> seen = new HashSet<>();
        List<Tensor> gradients = new ArrayList<>();
        try {
            double norm = 0;
            for (Parameter parameter : parameters) {
                try (Tensor value = parameter.value()) {
                    if (seen.add(value.id()) && value.hasGradient()) {
                        Tensor gradient = value.gradient();
                        gradients.add(gradient);
                        requireFinite(gradient);
                        try (Tensor magnitude = Ops.norm(gradient)) {
                            double local = magnitude.item();
                            if (Double.isInfinite(local)) {
                                local = 0;
                                for (float component : gradient.toFloatArray()) {
                                    local = Math.hypot(local, component);
                                }
                            }
                            norm = Math.hypot(norm, local);
                        }
                    }
                }
            }
            float factor = (float) Math.min(1, maximum / norm);
            if (factor < 1) {
                for (Tensor gradient : gradients) {
                    try (Tensor scaled = Ops.scale(gradient, factor)) {
                        gradient.context().assign(gradient, scaled);
                    }
                }
            }
            return norm;
        } finally {
            gradients.forEach(Tensor::close);
        }
    }
}

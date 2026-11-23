package org.dynamisengine.nn.autograd;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import org.dynamisengine.nn.ops.Ops;
import org.dynamisengine.nn.runtime.ExecutionContext;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.tensor.Shape;
import org.dynamisengine.nn.tensor.Tensor;
import org.dynamisengine.nn.tensor.TensorAccess;

/** Scoped, single-use reverse-mode tape. Backward retains accumulated leaf gradients. */
public final class GradientTape implements AutoCloseable {
    private static final AtomicLong IDS = new AtomicLong();
    private final long id = IDS.incrementAndGet();
    private final ExecutionContext context;
    private final List<Node> nodes = new ArrayList<>();
    private boolean closed;

    public GradientTape(ExecutionContext context) {
        this.context = context;
        context.attachTape(this);
    }

    /** Used by Ops before allocating a result. */
    public boolean watches(List<Tensor> inputs) {
        ensureRecording();
        boolean watches = false;
        for (Tensor input : inputs) {
            context.requireContext(input);
            if (input.requiresGrad()) {
                long owner = TensorAccess.tapeId(input);
                if (owner != 0 && owner != id) {
                    throw new IllegalStateException("Tensor belongs to another tape; detach it before reuse");
                }
                watches = true;
            }
        }
        return watches;
    }

    public void record(Tensor output, List<Tensor> inputs, BackwardRule backward) {
        if (!watches(inputs)) {
            return;
        }
        context.requireContext(output);
        if (output.dtype() != DType.FP32 || output.requiresGrad()) {
            throw new IllegalArgumentException("A recorded operation needs a fresh FP32 result");
        }
        List<TensorAccess.Saved> saved = new ArrayList<>();
        try {
            for (Tensor input : inputs) {
                saved.add(TensorAccess.save(input));
            }
            TensorAccess.track(output, id);
            nodes.add(new Node(output.retain(), List.copyOf(saved), backward));
        } catch (RuntimeException error) {
            saved.forEach(TensorAccess.Saved::close);
            throw error;
        }
    }

    public void backward(Tensor loss) {
        if (!loss.shape().equals(Shape.of())) {
            throw new IllegalArgumentException("Non-scalar outputs require an explicit upstream gradient");
        }
        try (Tensor seed = context.full(Shape.of(), 1f)) {
            backward(loss, seed);
        }
    }

    public void backward(Tensor output, Tensor upstream) {
        ensureRecording();
        context.requireContext(output);
        context.requireContext(upstream);
        if (!output.requiresGrad() || !output.shape().equals(upstream.shape())
                || output.dtype() != upstream.dtype()) {
            throw new IllegalArgumentException("Backward requires tracked output and a matching upstream gradient");
        }
        watches(List.of(output));
        context.detachTape(this);
        Map<Long, Tensor> gradients = new LinkedHashMap<>();
        Map<Long, Tensor> leaves = new LinkedHashMap<>();
        try {
            gradients.put(output.id(), Ops.copy(upstream));
            if (output.isLeaf()) {
                leaves.put(output.id(), output);
            }
            for (Node node : nodes) {
                for (TensorAccess.Saved saved : node.inputs) {
                    Tensor value = saved.tensor();
                    if (value.isLeaf()) {
                        leaves.putIfAbsent(value.id(), value);
                    }
                }
            }
            for (int i = nodes.size() - 1; i >= 0; i--) {
                Node node = nodes.get(i);
                Tensor gradient = gradients.remove(node.output.id());
                if (gradient == null) {
                    continue;
                }
                try (gradient) {
                    List<Tensor> inputs = node.inputs.stream().map(TensorAccess.Saved::tensor).toList();
                    Tensor[] contributions = node.backward.apply(gradient, inputs);
                    try {
                        if (contributions.length != inputs.size()) {
                            throw new IllegalStateException("Backward rule returned the wrong number of gradients");
                        }
                        for (int j = 0; j < inputs.size(); j++) {
                            Tensor input = inputs.get(j);
                            Tensor contribution = contributions[j];
                            if (!input.requiresGrad() || contribution == null) {
                                continue;
                            }
                            context.requireContext(contribution);
                            if (!input.shape().equals(contribution.shape()) || input.dtype() != contribution.dtype()) {
                                throw new IllegalStateException("Backward rule returned an invalid input gradient");
                            }
                            Tensor previous = gradients.get(input.id());
                            Tensor combined = previous == null ? contribution.retain() : Ops.add(previous, contribution);
                            gradients.put(input.id(), combined);
                            if (previous != null) {
                                previous.close();
                            }
                        }
                    } finally {
                        for (Tensor contribution : contributions) {
                            if (contribution != null) {
                                contribution.close();
                            }
                        }
                    }
                }
            }
            // Observe deferred failures before committing any new leaf gradients.
            context.synchronize();
            for (Map.Entry<Long, Tensor> entry : leaves.entrySet()) {
                Tensor gradient = gradients.get(entry.getKey());
                if (gradient != null) {
                    Tensor leaf = entry.getValue();
                    if (leaf.hasGradient()) {
                        try (Tensor previous = leaf.gradient(); Tensor combined = Ops.add(previous, gradient)) {
                            TensorAccess.setGradient(leaf, combined);
                        }
                    } else {
                        TensorAccess.setGradient(leaf, gradient);
                    }
                }
            }
        } finally {
            gradients.values().forEach(Tensor::close);
            close();
        }
    }

    private void ensureRecording() {
        context.ensureOpen();
        if (closed || context.activeTape() != this) {
            throw new IllegalStateException("Gradient tape is closed or already consumed");
        }
    }

    @Override
    public void close() {
        context.checkThread();
        if (!closed) {
            closed = true;
            context.detachTape(this);
            for (Node node : nodes) {
                node.inputs.forEach(TensorAccess.Saved::close);
                node.output.close();
            }
            nodes.clear();
        }
    }

    private record Node(Tensor output, List<TensorAccess.Saved> inputs, BackwardRule backward) {
    }
}

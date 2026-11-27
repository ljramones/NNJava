package org.dynamisengine.nn.data;

import org.dynamisengine.nn.runtime.ExecutionContext;

import java.util.Arrays;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Random;

/**
 * Stateful deterministic batching; borrows the dataset and owns its ordering state.
 */
public final class DataLoader implements AutoCloseable {
    private final Dataset dataset;
    private final ExecutionContext context;
    private final int batchSize, size;
    private final boolean shuffle, dropLast;
    private final long seed;
    private int[] order;
    private long epoch;
    private int cursor;
    private boolean closed;
    public DataLoader(ExecutionContext context, Dataset dataset, int batchSize, boolean shuffle, boolean dropLast, long seed) {
        if (batchSize <= 0 || dataset.size() > Integer.MAX_VALUE - 8L) {
            throw new IllegalArgumentException("Invalid loader size/batch size");
        }
        this.context = context;
        this.dataset = dataset;
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.dropLast = dropLast;
        this.seed = seed;
        size = Math.toIntExact(dataset.size());
        order = ordering(0);
    }

    private int[] ordering(long epoch) {
        int[] result = new int[size];
        for (int i = 0; i < size; i++) {
            result[i] = i;
        }
        if (shuffle) {
            Random random = new Random(seed ^ Long.rotateLeft(epoch * 0x9e3779b97f4a7c15L, 17));
            for (int i = size - 1; i > 0; i--) {
                int j = random.nextInt(i + 1);
                int old = result[i];
                result[i] = result[j];
                result[j] = old;
            }
        }
        return result;
    }

    public boolean hasNext() {
        ensureOpen();
        return cursor < size && (!dropLast || size - cursor >= batchSize);
    }

    public Batch next() {
        if (!hasNext()) {
            throw new NoSuchElementException("Epoch exhausted");
        }
        int end = (int) Math.min(size, (long) cursor + batchSize);
        Batch result = dataset.batch(context, Arrays.copyOfRange(order, cursor, end));
        cursor = end;
        return result;
    }

    public void nextEpoch() {
        ensureOpen();
        long next = Math.addExact(epoch, 1);
        int[] nextOrder = ordering(next);
        epoch = next;
        cursor = 0;
        order = nextOrder;
    }

    public State state() {
        ensureOpen();
        return new State(seed, epoch, cursor, batchSize, shuffle, dropLast, size);
    }

    public void restore(State state) {
        ensureOpen();
        if (state.seed != seed || state.batchSize != batchSize || state.shuffle != shuffle || state.dropLast != dropLast || state.size != size
                || state.epoch < 0 || state.cursor < 0 || state.cursor > size || state.cursor != size && state.cursor % batchSize != 0) {
            throw new IllegalArgumentException("Loader state/configuration mismatch");
        }
        int[] restored = ordering(state.epoch);
        epoch = state.epoch;
        cursor = state.cursor;
        order = restored;
    }

    private void ensureOpen() {
        context.ensureOpen();
        if (closed) {
            throw new IllegalStateException("Loader is closed");
        }
    }

    @Override
    public void close() {
        closed = true;
    }

    public record State(long seed, long epoch, int cursor, int batchSize, boolean shuffle, boolean dropLast, int size) {
        public static State fromMetadata(Map<String, String> values) {
            return new State(Long.parseLong(values.get("loader.seed")), Long.parseLong(values.get("loader.epoch")), Integer.parseInt(values.get("loader.cursor")),
                    Integer.parseInt(values.get("loader.batchSize")), Boolean.parseBoolean(values.get("loader.shuffle")), Boolean.parseBoolean(values.get("loader.dropLast")), Integer.parseInt(values.get("loader.size")));
        }

        public Map<String, String> metadata() {
            return Map.of("loader.seed", Long.toString(seed), "loader.epoch", Long.toString(epoch), "loader.cursor", Integer.toString(cursor),
                    "loader.batchSize", Integer.toString(batchSize), "loader.shuffle", Boolean.toString(shuffle), "loader.dropLast", Boolean.toString(dropLast), "loader.size", Integer.toString(size));
        }
    }
}

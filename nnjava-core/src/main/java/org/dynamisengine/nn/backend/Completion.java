package org.dynamisengine.nn.backend;

/**
 * Completion of ordered backend work. Await must also report execution failures.
 */
public interface Completion {
    static Completion completed() {
        return new Completion() {
            @Override
            public boolean isDone() {
                return true;
            }

            @Override
            public void await() {
            }
        };
    }

    boolean isDone();

    void await();
}

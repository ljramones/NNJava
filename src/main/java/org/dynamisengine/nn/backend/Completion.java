package org.dynamisengine.nn.backend;

/** Completion of ordered backend work. Await must also report execution failures. */
public interface Completion {
    boolean isDone();
    void await();

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
}

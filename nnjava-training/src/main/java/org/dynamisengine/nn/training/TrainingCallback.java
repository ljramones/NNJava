package org.dynamisengine.nn.training;

/** Called synchronously on the context thread after a committed training batch. */
@FunctionalInterface
public interface TrainingCallback {
    Trainer.Decision onEvent(Trainer.Event event);
}

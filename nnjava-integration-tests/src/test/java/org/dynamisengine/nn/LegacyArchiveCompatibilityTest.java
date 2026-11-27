package org.dynamisengine.nn;

import org.dynamisengine.nn.io.StateArchive;
import org.dynamisengine.nn.tensor.DType;
import org.dynamisengine.nn.training.Checkpoint;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

class LegacyArchiveCompatibilityTest {
    @TempDir
    Path temporary;

    @Test
    void readsCheckpointProducedByTheOriginalMonolithicArtifact() throws Exception {
        Path file = temporary.resolve("original.nnj");
        try (var input = getClass().getResourceAsStream("/legacy/checkpoint-v1.nnj")) {
            assertNotNull(input);
            Files.copy(input, file);
        }
        var archive = StateArchive.read(file);
        assertArrayEquals(new float[]{1.25f, -2.5f}, archive.model().tensors().get("weights").floats());
        assertArrayEquals(new int[]{7, -3}, archive.model().tensors().get("indices").integers());
        assertArrayEquals(new boolean[]{true, false}, archive.model().tensors().get("mask").booleans());
        assertEquals(DType.BOOL, archive.model().tensors().get("mask").dtype());
        assertEquals("weights", archive.model().aliases().get("tied"));
        assertEquals(17L, archive.model().trainingState().get("dropout.counter"));
        assertEquals("legacy", archive.model().configuration().get("format"));
        assertEquals("monolithic-v1", archive.metadata().get("source"));
        var checkpoint = Checkpoint.read(file);
        assertEquals(9L, checkpoint.optimizer().counters().get("steps"));
        assertArrayEquals(new float[]{0.5f, -0.25f}, checkpoint.optimizer().tensors().get("momentum").floats());
        Path rewritten = temporary.resolve("rewritten.nnj");
        Checkpoint.write(rewritten, checkpoint);
        assertArrayEquals(Files.readAllBytes(file), Files.readAllBytes(rewritten));
    }
}

package org.dynamisengine.nn.nn;

/** Whether a recurrent sequence returns [batch,time,hidden] or the last valid [batch,hidden]. */
public enum SequenceOutput { ALL, LAST }

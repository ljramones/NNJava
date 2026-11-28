package org.dynamisengine.nn.generation;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import org.dynamisengine.nn.model.ComponentSpec;

/** Reusable standard generation settings; model and application hooks are bound separately. */
public record GenerationConfig(SamplingParameters sampling,int maxNewTokens,int minNewTokens,
                                Set<Integer> stopTokenIds,List<List<Integer>> stopSequences) {
    public GenerationConfig {
        Objects.requireNonNull(sampling); stopTokenIds=Set.copyOf(stopTokenIds);
        stopSequences=stopSequences.stream().map(List::copyOf).toList();
        if (maxNewTokens<0 || minNewTokens<0 || minNewTokens>maxNewTokens || stopTokenIds.stream().anyMatch(id -> id<0)
                || stopSequences.stream().anyMatch(ids -> ids.isEmpty() || ids.stream().anyMatch(id -> id<0))) {
            throw new IllegalArgumentException("Invalid generation budget or stop IDs/sequences");
        }
    }

    public ComponentSpec specification(String name) {
        var p=sampling;
        return new ComponentSpec("nnjava.generation",1,name,Map.of(
                "temperature",Float.toHexString(p.temperature()),"topK",Integer.toString(p.topK()),"topP",Float.toHexString(p.topP()),
                "repetitionPenalty",Float.toHexString(p.repetitionPenalty()),"presencePenalty",Float.toHexString(p.presencePenalty()),
                "frequencyPenalty",Float.toHexString(p.frequencyPenalty()),"seed",Long.toString(p.seed()),
                "maxNewTokens",Integer.toString(maxNewTokens),"minNewTokens",Integer.toString(minNewTokens),
                "stopTokenIds",encode(stopTokenIds.stream().sorted().toList())),
                java.util.stream.IntStream.range(0,stopSequences.size()).mapToObj(i -> new ComponentSpec("nnjava.token-stop",1,"stop"+i,
                        Map.of("ids",encode(stopSequences.get(i))),List.of())).toList());
    }

    public static GenerationConfig fromSpec(ComponentSpec spec) {
        spec.require("nnjava.generation",1,"temperature","topK","topP","repetitionPenalty","presencePenalty","frequencyPenalty","seed","maxNewTokens","minNewTokens","stopTokenIds");
        var sampling=new SamplingParameters(spec.floatValue("temperature"),spec.intValue("topK"),spec.floatValue("topP"),spec.floatValue("repetitionPenalty"),
                spec.floatValue("presencePenalty"),spec.floatValue("frequencyPenalty"),spec.longValue("seed"));
        var sequences=spec.children().stream().map(child -> {
            child.require("nnjava.token-stop",1,"ids");
            if (!child.children().isEmpty()) { throw new IllegalArgumentException("Token stop cannot have children"); }
            return decode(child.attribute("ids"));
        }).toList();
        var stops=decode(spec.attribute("stopTokenIds"));
        if (Set.copyOf(stops).size()!=stops.size()) { throw new IllegalArgumentException("Duplicate stop token IDs"); }
        return new GenerationConfig(sampling,spec.intValue("maxNewTokens"),spec.intValue("minNewTokens"),Set.copyOf(stops),sequences);
    }

    private static String encode(List<Integer> ids) { return ids.stream().map(String::valueOf).collect(Collectors.joining(",")); }
    private static List<Integer> decode(String text) { return text.isEmpty() ? List.of() : Arrays.stream(text.split(",",-1)).map(Integer::parseInt).toList(); }
}

package org.hse.musicrecognition.dto;

import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public record FingerprintCandidatesResponse(

        @JsonProperty("matched")
        @JsonAlias({"match", "matched"})
        Boolean matched,

        @JsonProperty("track_id")
        @JsonAlias({"track_id", "trackId"})
        Integer trackId,

        Double confidence,

        String reason,

        String source,

        @JsonProperty("top_k")
        @JsonAlias({"top_k", "topK"})
        Integer topK,

        @JsonProperty("query_hashes")
        @JsonAlias({"query_hashes", "queryHashes"})
        Integer queryHashes,

        @JsonProperty("unique_query_hashes")
        @JsonAlias({"unique_query_hashes", "uniqueQueryHashes"})
        Integer uniqueQueryHashes,

        @JsonProperty("best_aligned_matches")
        @JsonAlias({"best_aligned_matches", "bestAlignedMatches"})
        Integer bestAlignedMatches,

        @JsonProperty("second_aligned_matches")
        @JsonAlias({"second_aligned_matches", "secondAlignedMatches"})
        Integer secondAlignedMatches,

        List<FingerprintCandidateDto> candidates
) {
    public boolean isMatched() {
        return Boolean.TRUE.equals(matched);
    }

    public FingerprintCandidateDto bestCandidate() {
        if (candidates == null || candidates.isEmpty()) {
            return null;
        }

        if (trackId == null) {
            return candidates.get(0);
        }

        return candidates.stream()
                .filter(candidate -> trackId.equals(candidate.trackId()))
                .findFirst()
                .orElse(candidates.get(0));
    }
}
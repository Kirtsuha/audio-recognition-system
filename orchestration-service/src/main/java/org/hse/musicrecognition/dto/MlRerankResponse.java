package org.hse.musicrecognition.dto;

import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.util.List;
import java.util.Map;

@JsonIgnoreProperties(ignoreUnknown = true)
public record MlRerankResponse(

        @JsonAlias({"request_id", "requestId"})
        String requestId,

        String source,

        Boolean matched,

        String reason,

        RerankedCandidateDto best,

        List<RerankedCandidateDto> candidates,

        @JsonAlias({"timing_ms", "timingMs"})
        Map<String, Integer> timingMs
) {
    public boolean isMatched() {
        return Boolean.TRUE.equals(matched);
    }
}
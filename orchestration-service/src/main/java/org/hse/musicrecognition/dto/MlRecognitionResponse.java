package org.hse.musicrecognition.dto;

import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record MlRecognitionResponse(

        @JsonAlias({"requestId", "request_id"})
        String requestId,

        @JsonAlias({"trackId", "track_id", "songId", "song_id"})
        Integer trackId,

        @JsonAlias({"matched", "match"})
        Boolean matched,

        @JsonAlias({"confidence"})
        Double confidence,

        @JsonAlias({"score"})
        Double score,

        @JsonAlias({"margin"})
        Double margin,

        @JsonAlias({"support"})
        Integer support,

        @JsonAlias({"support_ratio", "supportRatio"})
        Double supportRatio,

        @JsonAlias({"reason"})
        String reason,

        @JsonAlias({"error"})
        String error
) {
}